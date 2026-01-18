use std::fmt::{self, Display};
use std::mem;
use std::os::raw::c_void;

use crate::array::VarLenArray;
use crate::string::{FixedAscii, FixedUnicode, VarLenAscii, VarLenUnicode};

#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(Copy, Clone)]
pub(crate) struct hvl_t {
    pub len: usize,
    pub ptr: *mut c_void,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum IntSize {
    U1 = 1,
    U2 = 2,
    U4 = 4,
    U8 = 8,
}

impl IntSize {
    pub const fn from_int(size: usize) -> Option<Self> {
        if size == 1 {
            Some(Self::U1)
        } else if size == 2 {
            Some(Self::U2)
        } else if size == 4 {
            Some(Self::U4)
        } else if size == 8 {
            Some(Self::U8)
        } else {
            None
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum FloatSize {
    #[cfg(feature = "f16")]
    U2 = 2,
    U4 = 4,
    U8 = 8,
}

impl FloatSize {
    pub const fn from_int(size: usize) -> Option<Self> {
        #[cfg(feature = "f16")]
        {
            if size == 2 {
                return Some(Self::U2);
            }
        }
        match size {
            4 => Some(Self::U4),
            8 => Some(Self::U8),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EnumMember {
    pub name: String,
    pub value: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EnumType {
    pub size: IntSize,
    pub signed: bool,
    pub members: Vec<EnumMember>,
}

impl EnumType {
    #[inline]
    pub fn base_type(&self) -> TypeDescriptor {
        if self.signed {
            TypeDescriptor::Integer(self.size)
        } else {
            TypeDescriptor::Unsigned(self.size)
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompoundField {
    pub name: String,
    pub ty: TypeDescriptor,
    pub offset: usize,
    pub index: usize,
}

impl CompoundField {
    pub fn new(name: &str, ty: TypeDescriptor, offset: usize, index: usize) -> Self {
        Self { name: name.to_owned(), ty, offset, index }
    }

    pub fn typed<T: H5Type>(name: &str, offset: usize, index: usize) -> Self {
        Self::new(name, T::type_descriptor(), offset, index)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompoundType {
    pub fields: Vec<CompoundField>,
    pub size: usize,
}

impl CompoundType {
    pub fn to_c_repr(&self) -> Self {
        let mut layout = self.clone();
        layout.fields.sort_by_key(|f| f.index);
        let mut offset = 0;
        let mut max_align = 1;
        for f in &mut layout.fields {
            f.ty = f.ty.to_c_repr();
            let align = f.ty.c_alignment();
            while offset % align != 0 {
                offset += 1;
            }
            f.offset = offset;
            max_align = max_align.max(align);
            offset += f.ty.size();
            layout.size = offset;
            while layout.size % max_align != 0 {
                layout.size += 1;
            }
        }
        layout
    }

    pub fn to_packed_repr(&self) -> Self {
        let mut layout = self.clone();
        layout.fields.sort_by_key(|f| f.index);
        layout.size = 0;
        for f in &mut layout.fields {
            f.ty = f.ty.to_packed_repr();
            f.offset = layout.size;
            layout.size += f.ty.size();
        }
        layout
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TypeDescriptor {
    Integer(IntSize),
    Unsigned(IntSize),
    Float(FloatSize),
    Boolean,
    Enum(EnumType),
    Compound(CompoundType),
    FixedArray(Box<Self>, usize),
    FixedAscii(usize),
    FixedUnicode(usize),
    VarLenArray(Box<Self>),
    VarLenAscii,
    VarLenUnicode,
}

impl Display for TypeDescriptor {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            TypeDescriptor::Integer(IntSize::U1) => write!(f, "int8"),
            TypeDescriptor::Integer(IntSize::U2) => write!(f, "int16"),
            TypeDescriptor::Integer(IntSize::U4) => write!(f, "int32"),
            TypeDescriptor::Integer(IntSize::U8) => write!(f, "int64"),
            TypeDescriptor::Unsigned(IntSize::U1) => write!(f, "uint8"),
            TypeDescriptor::Unsigned(IntSize::U2) => write!(f, "uint16"),
            TypeDescriptor::Unsigned(IntSize::U4) => write!(f, "uint32"),
            TypeDescriptor::Unsigned(IntSize::U8) => write!(f, "uint64"),
            #[cfg(feature = "f16")]
            TypeDescriptor::Float(FloatSize::U2) => write!(f, "float16"),
            TypeDescriptor::Float(FloatSize::U4) => write!(f, "float32"),
            TypeDescriptor::Float(FloatSize::U8) => write!(f, "float64"),
            TypeDescriptor::Boolean => write!(f, "bool"),
            TypeDescriptor::Enum(ref tp) => write!(f, "enum ({})", tp.base_type()),
            TypeDescriptor::Compound(ref tp) => write!(f, "compound ({} fields)", tp.fields.len()),
            TypeDescriptor::FixedArray(ref tp, n) => write!(f, "[{}; {}]", tp, n),
            TypeDescriptor::FixedAscii(n) => write!(f, "string (len {})", n),
            TypeDescriptor::FixedUnicode(n) => write!(f, "unicode (len {})", n),
            TypeDescriptor::VarLenArray(ref tp) => write!(f, "[{}] (var len)", tp),
            TypeDescriptor::VarLenAscii => write!(f, "string (var len)"),
            TypeDescriptor::VarLenUnicode => write!(f, "unicode (var len)"),
        }
    }
}

impl TypeDescriptor {
    pub fn size(&self) -> usize {
        match *self {
            Self::Integer(size) | Self::Unsigned(size) => size as _,
            Self::Float(size) => size as _,
            Self::Boolean => 1,
            Self::Enum(ref enum_type) => enum_type.size as _,
            Self::Compound(ref compound) => compound.size,
            Self::FixedArray(ref ty, len) => ty.size() * len,
            Self::FixedAscii(len) | Self::FixedUnicode(len) => len,
            Self::VarLenArray(_) => mem::size_of::<hvl_t>(),
            Self::VarLenAscii | Self::VarLenUnicode => mem::size_of::<*const u8>(),
        }
    }

    fn c_alignment(&self) -> usize {
        match *self {
            Self::Compound(ref compound) => {
                compound.fields.iter().map(|f| f.ty.c_alignment()).max().unwrap_or(1)
            }
            Self::FixedArray(ref ty, _) => ty.c_alignment(),
            Self::FixedAscii(_) | Self::FixedUnicode(_) => 1,
            Self::VarLenArray(_) => mem::size_of::<usize>(),
            _ => self.size(),
        }
    }

    pub fn to_c_repr(&self) -> Self {
        match *self {
            Self::Compound(ref compound) => Self::Compound(compound.to_c_repr()),
            Self::FixedArray(ref ty, size) => Self::FixedArray(Box::new(ty.to_c_repr()), size),
            Self::VarLenArray(ref ty) => Self::VarLenArray(Box::new(ty.to_c_repr())),
            _ => self.clone(),
        }
    }

    pub fn to_packed_repr(&self) -> Self {
        match *self {
            Self::Compound(ref compound) => Self::Compound(compound.to_packed_repr()),
            Self::FixedArray(ref ty, size) => Self::FixedArray(Box::new(ty.to_packed_repr()), size),
            Self::VarLenArray(ref ty) => Self::VarLenArray(Box::new(ty.to_packed_repr())),
            _ => self.clone(),
        }
    }
}

/// Types that can be stored and retrieved from HDF5 datasets.
///
/// # Safety
///
/// Implementers must ensure that:
///
/// 1. **Accurate Type Descriptor**: The `type_descriptor()` must accurately represent
///    the memory layout of the type, including size, alignment, and field offsets.
///
/// 2. **Valid Memory Layout**: For compound types, all fields must have valid offsets
///    matching the type's actual memory layout. The type must have a `repr(C)` or
///    `repr(packed)` attribute to ensure consistent layout.
///
/// 3. **No Padding Issues**: The type must not have padding bytes that contain
///    uninitialized data when read from HDF5. All padding should be explicitly
///    initialized or the type should use `repr(packed)`.
///
/// 4. **Copy Safety**: The type must be `Copy` or must be safely copyable byte-for-byte.
///    Types with custom `Drop` implementations or self-referential types must not
///    implement this trait.
///
/// 5. **Enum Discriminants**: For enum types, the discriminant values must match
///    the values described in the type descriptor.
///
/// Failure to uphold these invariants may result in undefined behavior, including
/// memory corruption and segmentation faults.
pub unsafe trait H5Type: 'static {
    /// Returns the type descriptor for this type.
    ///
    /// The descriptor must accurately describe the memory layout of the type.
    fn type_descriptor() -> TypeDescriptor;
}

macro_rules! impl_h5type {
    ($ty:ty, $variant:ident, $size:expr) => {
        unsafe impl H5Type for $ty {
            #[inline]
            fn type_descriptor() -> TypeDescriptor {
                $crate::h5type::TypeDescriptor::$variant($size)
            }
        }
    };
}

impl_h5type!(i8, Integer, IntSize::U1);
impl_h5type!(i16, Integer, IntSize::U2);
impl_h5type!(i32, Integer, IntSize::U4);
impl_h5type!(i64, Integer, IntSize::U8);
impl_h5type!(u8, Unsigned, IntSize::U1);
impl_h5type!(u16, Unsigned, IntSize::U2);
impl_h5type!(u32, Unsigned, IntSize::U4);
impl_h5type!(u64, Unsigned, IntSize::U8);
#[cfg(feature = "f16")]
impl_h5type!(::half::f16, Float, FloatSize::U2);
impl_h5type!(f32, Float, FloatSize::U4);
impl_h5type!(f64, Float, FloatSize::U8);

#[cfg(target_pointer_width = "32")]
impl_h5type!(isize, Integer, IntSize::U4);
#[cfg(target_pointer_width = "32")]
impl_h5type!(usize, Unsigned, IntSize::U4);

#[cfg(target_pointer_width = "64")]
impl_h5type!(isize, Integer, IntSize::U8);
#[cfg(target_pointer_width = "64")]
impl_h5type!(usize, Unsigned, IntSize::U8);

unsafe impl H5Type for bool {
    #[inline]
    fn type_descriptor() -> TypeDescriptor {
        TypeDescriptor::Boolean
    }
}

macro_rules! impl_tuple {
    // Single element tuple
    ($t:ident) => (
        unsafe impl<$t> H5Type for ($t,) where $t: H5Type {
            #[inline]
            fn type_descriptor() -> TypeDescriptor {
                let size = mem::size_of::<($t,)>();
                assert_eq!(size, mem::size_of::<$t>());
                TypeDescriptor::Compound(CompoundType {
                    fields: vec![CompoundField::typed::<$t>("0", 0, 0)],
                    size,
                })
            }
        }
    );

    // Multi-element tuples - delegate to impl_tuple_n
    ($t:ident, $($tt:ident),*) => (
        impl_tuple_n!([$t, $($tt),*] 0);
        impl_tuple!($($tt),*);
    );
}

// Helper macro to implement H5Type for N-tuples using offset_of!
macro_rules! impl_tuple_n {
    // 2-tuple
    ([$t0:ident, $t1:ident] $($_idx:tt)*) => {
        #[allow(dead_code, unused_variables)]
        unsafe impl<$t0, $t1> H5Type for ($t0, $t1)
            where $t0: H5Type, $t1: H5Type
        {
            fn type_descriptor() -> TypeDescriptor {
                let mut fields = vec![
                    CompoundField {
                        name: "0".to_string(),
                        ty: <$t0 as H5Type>::type_descriptor(),
                        offset: mem::offset_of!(Self, 0),
                        index: 0,
                    },
                    CompoundField {
                        name: "1".to_string(),
                        ty: <$t1 as H5Type>::type_descriptor(),
                        offset: mem::offset_of!(Self, 1),
                        index: 1,
                    },
                ];
                let size = mem::size_of::<Self>();
                fields.sort_by_key(|f| f.offset);
                TypeDescriptor::Compound(CompoundType { fields, size })
            }
        }
    };
    // 3-tuple
    ([$t0:ident, $t1:ident, $t2:ident] $($_idx:tt)*) => {
        #[allow(dead_code, unused_variables)]
        unsafe impl<$t0, $t1, $t2> H5Type for ($t0, $t1, $t2)
            where $t0: H5Type, $t1: H5Type, $t2: H5Type
        {
            fn type_descriptor() -> TypeDescriptor {
                let mut fields = vec![
                    CompoundField {
                        name: "0".to_string(),
                        ty: <$t0 as H5Type>::type_descriptor(),
                        offset: mem::offset_of!(Self, 0),
                        index: 0,
                    },
                    CompoundField {
                        name: "1".to_string(),
                        ty: <$t1 as H5Type>::type_descriptor(),
                        offset: mem::offset_of!(Self, 1),
                        index: 1,
                    },
                    CompoundField {
                        name: "2".to_string(),
                        ty: <$t2 as H5Type>::type_descriptor(),
                        offset: mem::offset_of!(Self, 2),
                        index: 2,
                    },
                ];
                let size = mem::size_of::<Self>();
                fields.sort_by_key(|f| f.offset);
                TypeDescriptor::Compound(CompoundType { fields, size })
            }
        }
    };
    // 4-tuple
    ([$t0:ident, $t1:ident, $t2:ident, $t3:ident] $($_idx:tt)*) => {
        #[allow(dead_code, unused_variables)]
        unsafe impl<$t0, $t1, $t2, $t3> H5Type for ($t0, $t1, $t2, $t3)
            where $t0: H5Type, $t1: H5Type, $t2: H5Type, $t3: H5Type
        {
            fn type_descriptor() -> TypeDescriptor {
                let mut fields = vec![
                    CompoundField {
                        name: "0".to_string(),
                        ty: <$t0 as H5Type>::type_descriptor(),
                        offset: mem::offset_of!(Self, 0),
                        index: 0,
                    },
                    CompoundField {
                        name: "1".to_string(),
                        ty: <$t1 as H5Type>::type_descriptor(),
                        offset: mem::offset_of!(Self, 1),
                        index: 1,
                    },
                    CompoundField {
                        name: "2".to_string(),
                        ty: <$t2 as H5Type>::type_descriptor(),
                        offset: mem::offset_of!(Self, 2),
                        index: 2,
                    },
                    CompoundField {
                        name: "3".to_string(),
                        ty: <$t3 as H5Type>::type_descriptor(),
                        offset: mem::offset_of!(Self, 3),
                        index: 3,
                    },
                ];
                let size = mem::size_of::<Self>();
                fields.sort_by_key(|f| f.offset);
                TypeDescriptor::Compound(CompoundType { fields, size })
            }
        }
    };
    // 5-tuple
    ([$t0:ident, $t1:ident, $t2:ident, $t3:ident, $t4:ident] $($_idx:tt)*) => {
        #[allow(dead_code, unused_variables)]
        unsafe impl<$t0, $t1, $t2, $t3, $t4> H5Type for ($t0, $t1, $t2, $t3, $t4)
            where $t0: H5Type, $t1: H5Type, $t2: H5Type, $t3: H5Type, $t4: H5Type
        {
            fn type_descriptor() -> TypeDescriptor {
                let mut fields = vec![
                    CompoundField {
                        name: "0".to_string(),
                        ty: <$t0 as H5Type>::type_descriptor(),
                        offset: mem::offset_of!(Self, 0),
                        index: 0,
                    },
                    CompoundField {
                        name: "1".to_string(),
                        ty: <$t1 as H5Type>::type_descriptor(),
                        offset: mem::offset_of!(Self, 1),
                        index: 1,
                    },
                    CompoundField {
                        name: "2".to_string(),
                        ty: <$t2 as H5Type>::type_descriptor(),
                        offset: mem::offset_of!(Self, 2),
                        index: 2,
                    },
                    CompoundField {
                        name: "3".to_string(),
                        ty: <$t3 as H5Type>::type_descriptor(),
                        offset: mem::offset_of!(Self, 3),
                        index: 3,
                    },
                    CompoundField {
                        name: "4".to_string(),
                        ty: <$t4 as H5Type>::type_descriptor(),
                        offset: mem::offset_of!(Self, 4),
                        index: 4,
                    },
                ];
                let size = mem::size_of::<Self>();
                fields.sort_by_key(|f| f.offset);
                TypeDescriptor::Compound(CompoundType { fields, size })
            }
        }
    };
    // 6-tuple
    ([$t0:ident, $t1:ident, $t2:ident, $t3:ident, $t4:ident, $t5:ident] $($_idx:tt)*) => {
        #[allow(dead_code, unused_variables)]
        unsafe impl<$t0, $t1, $t2, $t3, $t4, $t5> H5Type for ($t0, $t1, $t2, $t3, $t4, $t5)
            where $t0: H5Type, $t1: H5Type, $t2: H5Type, $t3: H5Type, $t4: H5Type, $t5: H5Type
        {
            fn type_descriptor() -> TypeDescriptor {
                let mut fields = vec![
                    CompoundField {
                        name: "0".to_string(),
                        ty: <$t0 as H5Type>::type_descriptor(),
                        offset: mem::offset_of!(Self, 0),
                        index: 0,
                    },
                    CompoundField {
                        name: "1".to_string(),
                        ty: <$t1 as H5Type>::type_descriptor(),
                        offset: mem::offset_of!(Self, 1),
                        index: 1,
                    },
                    CompoundField {
                        name: "2".to_string(),
                        ty: <$t2 as H5Type>::type_descriptor(),
                        offset: mem::offset_of!(Self, 2),
                        index: 2,
                    },
                    CompoundField {
                        name: "3".to_string(),
                        ty: <$t3 as H5Type>::type_descriptor(),
                        offset: mem::offset_of!(Self, 3),
                        index: 3,
                    },
                    CompoundField {
                        name: "4".to_string(),
                        ty: <$t4 as H5Type>::type_descriptor(),
                        offset: mem::offset_of!(Self, 4),
                        index: 4,
                    },
                    CompoundField {
                        name: "5".to_string(),
                        ty: <$t5 as H5Type>::type_descriptor(),
                        offset: mem::offset_of!(Self, 5),
                        index: 5,
                    },
                ];
                let size = mem::size_of::<Self>();
                fields.sort_by_key(|f| f.offset);
                TypeDescriptor::Compound(CompoundType { fields, size })
            }
        }
    };
    // 7-tuple
    ([$t0:ident, $t1:ident, $t2:ident, $t3:ident, $t4:ident, $t5:ident, $t6:ident] $($_idx:tt)*) => {
        #[allow(dead_code, unused_variables)]
        unsafe impl<$t0, $t1, $t2, $t3, $t4, $t5, $t6> H5Type for ($t0, $t1, $t2, $t3, $t4, $t5, $t6)
            where $t0: H5Type, $t1: H5Type, $t2: H5Type, $t3: H5Type, $t4: H5Type, $t5: H5Type, $t6: H5Type
        {
            fn type_descriptor() -> TypeDescriptor {
                let mut fields = vec![
                    CompoundField {
                        name: "0".to_string(),
                        ty: <$t0 as H5Type>::type_descriptor(),
                        offset: mem::offset_of!(Self, 0),
                        index: 0,
                    },
                    CompoundField {
                        name: "1".to_string(),
                        ty: <$t1 as H5Type>::type_descriptor(),
                        offset: mem::offset_of!(Self, 1),
                        index: 1,
                    },
                    CompoundField {
                        name: "2".to_string(),
                        ty: <$t2 as H5Type>::type_descriptor(),
                        offset: mem::offset_of!(Self, 2),
                        index: 2,
                    },
                    CompoundField {
                        name: "3".to_string(),
                        ty: <$t3 as H5Type>::type_descriptor(),
                        offset: mem::offset_of!(Self, 3),
                        index: 3,
                    },
                    CompoundField {
                        name: "4".to_string(),
                        ty: <$t4 as H5Type>::type_descriptor(),
                        offset: mem::offset_of!(Self, 4),
                        index: 4,
                    },
                    CompoundField {
                        name: "5".to_string(),
                        ty: <$t5 as H5Type>::type_descriptor(),
                        offset: mem::offset_of!(Self, 5),
                        index: 5,
                    },
                    CompoundField {
                        name: "6".to_string(),
                        ty: <$t6 as H5Type>::type_descriptor(),
                        offset: mem::offset_of!(Self, 6),
                        index: 6,
                    },
                ];
                let size = mem::size_of::<Self>();
                fields.sort_by_key(|f| f.offset);
                TypeDescriptor::Compound(CompoundType { fields, size })
            }
        }
    };
    // 8-tuple
    ([$t0:ident, $t1:ident, $t2:ident, $t3:ident, $t4:ident, $t5:ident, $t6:ident, $t7:ident] $($_idx:tt)*) => {
        #[allow(dead_code, unused_variables)]
        unsafe impl<$t0, $t1, $t2, $t3, $t4, $t5, $t6, $t7> H5Type for ($t0, $t1, $t2, $t3, $t4, $t5, $t6, $t7)
            where $t0: H5Type, $t1: H5Type, $t2: H5Type, $t3: H5Type, $t4: H5Type, $t5: H5Type, $t6: H5Type, $t7: H5Type
        {
            fn type_descriptor() -> TypeDescriptor {
                let mut fields = vec![
                    CompoundField {
                        name: "0".to_string(),
                        ty: <$t0 as H5Type>::type_descriptor(),
                        offset: mem::offset_of!(Self, 0),
                        index: 0,
                    },
                    CompoundField {
                        name: "1".to_string(),
                        ty: <$t1 as H5Type>::type_descriptor(),
                        offset: mem::offset_of!(Self, 1),
                        index: 1,
                    },
                    CompoundField {
                        name: "2".to_string(),
                        ty: <$t2 as H5Type>::type_descriptor(),
                        offset: mem::offset_of!(Self, 2),
                        index: 2,
                    },
                    CompoundField {
                        name: "3".to_string(),
                        ty: <$t3 as H5Type>::type_descriptor(),
                        offset: mem::offset_of!(Self, 3),
                        index: 3,
                    },
                    CompoundField {
                        name: "4".to_string(),
                        ty: <$t4 as H5Type>::type_descriptor(),
                        offset: mem::offset_of!(Self, 4),
                        index: 4,
                    },
                    CompoundField {
                        name: "5".to_string(),
                        ty: <$t5 as H5Type>::type_descriptor(),
                        offset: mem::offset_of!(Self, 5),
                        index: 5,
                    },
                    CompoundField {
                        name: "6".to_string(),
                        ty: <$t6 as H5Type>::type_descriptor(),
                        offset: mem::offset_of!(Self, 6),
                        index: 6,
                    },
                    CompoundField {
                        name: "7".to_string(),
                        ty: <$t7 as H5Type>::type_descriptor(),
                        offset: mem::offset_of!(Self, 7),
                        index: 7,
                    },
                ];
                let size = mem::size_of::<Self>();
                fields.sort_by_key(|f| f.offset);
                TypeDescriptor::Compound(CompoundType { fields, size })
            }
        }
    };
    // 9-tuple
    ([$t0:ident, $t1:ident, $t2:ident, $t3:ident, $t4:ident, $t5:ident, $t6:ident, $t7:ident, $t8:ident] $($_idx:tt)*) => {
        #[allow(dead_code, unused_variables)]
        unsafe impl<$t0, $t1, $t2, $t3, $t4, $t5, $t6, $t7, $t8> H5Type for ($t0, $t1, $t2, $t3, $t4, $t5, $t6, $t7, $t8)
            where $t0: H5Type, $t1: H5Type, $t2: H5Type, $t3: H5Type, $t4: H5Type, $t5: H5Type, $t6: H5Type, $t7: H5Type, $t8: H5Type
        {
            fn type_descriptor() -> TypeDescriptor {
                let mut fields = vec![
                    CompoundField {
                        name: "0".to_string(),
                        ty: <$t0 as H5Type>::type_descriptor(),
                        offset: mem::offset_of!(Self, 0),
                        index: 0,
                    },
                    CompoundField {
                        name: "1".to_string(),
                        ty: <$t1 as H5Type>::type_descriptor(),
                        offset: mem::offset_of!(Self, 1),
                        index: 1,
                    },
                    CompoundField {
                        name: "2".to_string(),
                        ty: <$t2 as H5Type>::type_descriptor(),
                        offset: mem::offset_of!(Self, 2),
                        index: 2,
                    },
                    CompoundField {
                        name: "3".to_string(),
                        ty: <$t3 as H5Type>::type_descriptor(),
                        offset: mem::offset_of!(Self, 3),
                        index: 3,
                    },
                    CompoundField {
                        name: "4".to_string(),
                        ty: <$t4 as H5Type>::type_descriptor(),
                        offset: mem::offset_of!(Self, 4),
                        index: 4,
                    },
                    CompoundField {
                        name: "5".to_string(),
                        ty: <$t5 as H5Type>::type_descriptor(),
                        offset: mem::offset_of!(Self, 5),
                        index: 5,
                    },
                    CompoundField {
                        name: "6".to_string(),
                        ty: <$t6 as H5Type>::type_descriptor(),
                        offset: mem::offset_of!(Self, 6),
                        index: 6,
                    },
                    CompoundField {
                        name: "7".to_string(),
                        ty: <$t7 as H5Type>::type_descriptor(),
                        offset: mem::offset_of!(Self, 7),
                        index: 7,
                    },
                    CompoundField {
                        name: "8".to_string(),
                        ty: <$t8 as H5Type>::type_descriptor(),
                        offset: mem::offset_of!(Self, 8),
                        index: 8,
                    },
                ];
                let size = mem::size_of::<Self>();
                fields.sort_by_key(|f| f.offset);
                TypeDescriptor::Compound(CompoundType { fields, size })
            }
        }
    };
    // 10-tuple
    ([$t0:ident, $t1:ident, $t2:ident, $t3:ident, $t4:ident, $t5:ident, $t6:ident, $t7:ident, $t8:ident, $t9:ident] $($_idx:tt)*) => {
        #[allow(dead_code, unused_variables)]
        unsafe impl<$t0, $t1, $t2, $t3, $t4, $t5, $t6, $t7, $t8, $t9> H5Type for ($t0, $t1, $t2, $t3, $t4, $t5, $t6, $t7, $t8, $t9)
            where $t0: H5Type, $t1: H5Type, $t2: H5Type, $t3: H5Type, $t4: H5Type, $t5: H5Type, $t6: H5Type, $t7: H5Type, $t8: H5Type, $t9: H5Type
        {
            fn type_descriptor() -> TypeDescriptor {
                let mut fields = vec![
                    CompoundField {
                        name: "0".to_string(),
                        ty: <$t0 as H5Type>::type_descriptor(),
                        offset: mem::offset_of!(Self, 0),
                        index: 0,
                    },
                    CompoundField {
                        name: "1".to_string(),
                        ty: <$t1 as H5Type>::type_descriptor(),
                        offset: mem::offset_of!(Self, 1),
                        index: 1,
                    },
                    CompoundField {
                        name: "2".to_string(),
                        ty: <$t2 as H5Type>::type_descriptor(),
                        offset: mem::offset_of!(Self, 2),
                        index: 2,
                    },
                    CompoundField {
                        name: "3".to_string(),
                        ty: <$t3 as H5Type>::type_descriptor(),
                        offset: mem::offset_of!(Self, 3),
                        index: 3,
                    },
                    CompoundField {
                        name: "4".to_string(),
                        ty: <$t4 as H5Type>::type_descriptor(),
                        offset: mem::offset_of!(Self, 4),
                        index: 4,
                    },
                    CompoundField {
                        name: "5".to_string(),
                        ty: <$t5 as H5Type>::type_descriptor(),
                        offset: mem::offset_of!(Self, 5),
                        index: 5,
                    },
                    CompoundField {
                        name: "6".to_string(),
                        ty: <$t6 as H5Type>::type_descriptor(),
                        offset: mem::offset_of!(Self, 6),
                        index: 6,
                    },
                    CompoundField {
                        name: "7".to_string(),
                        ty: <$t7 as H5Type>::type_descriptor(),
                        offset: mem::offset_of!(Self, 7),
                        index: 7,
                    },
                    CompoundField {
                        name: "8".to_string(),
                        ty: <$t8 as H5Type>::type_descriptor(),
                        offset: mem::offset_of!(Self, 8),
                        index: 8,
                    },
                    CompoundField {
                        name: "9".to_string(),
                        ty: <$t9 as H5Type>::type_descriptor(),
                        offset: mem::offset_of!(Self, 9),
                        index: 9,
                    },
                ];
                let size = mem::size_of::<Self>();
                fields.sort_by_key(|f| f.offset);
                TypeDescriptor::Compound(CompoundType { fields, size })
            }
        }
    };
    // 11-tuple
    ([$t0:ident, $t1:ident, $t2:ident, $t3:ident, $t4:ident, $t5:ident, $t6:ident, $t7:ident, $t8:ident, $t9:ident, $t10:ident] $($_idx:tt)*) => {
        #[allow(dead_code, unused_variables)]
        unsafe impl<$t0, $t1, $t2, $t3, $t4, $t5, $t6, $t7, $t8, $t9, $t10> H5Type for ($t0, $t1, $t2, $t3, $t4, $t5, $t6, $t7, $t8, $t9, $t10)
            where $t0: H5Type, $t1: H5Type, $t2: H5Type, $t3: H5Type, $t4: H5Type, $t5: H5Type, $t6: H5Type, $t7: H5Type, $t8: H5Type, $t9: H5Type, $t10: H5Type
        {
            fn type_descriptor() -> TypeDescriptor {
                let mut fields = vec![
                    CompoundField {
                        name: "0".to_string(),
                        ty: <$t0 as H5Type>::type_descriptor(),
                        offset: mem::offset_of!(Self, 0),
                        index: 0,
                    },
                    CompoundField {
                        name: "1".to_string(),
                        ty: <$t1 as H5Type>::type_descriptor(),
                        offset: mem::offset_of!(Self, 1),
                        index: 1,
                    },
                    CompoundField {
                        name: "2".to_string(),
                        ty: <$t2 as H5Type>::type_descriptor(),
                        offset: mem::offset_of!(Self, 2),
                        index: 2,
                    },
                    CompoundField {
                        name: "3".to_string(),
                        ty: <$t3 as H5Type>::type_descriptor(),
                        offset: mem::offset_of!(Self, 3),
                        index: 3,
                    },
                    CompoundField {
                        name: "4".to_string(),
                        ty: <$t4 as H5Type>::type_descriptor(),
                        offset: mem::offset_of!(Self, 4),
                        index: 4,
                    },
                    CompoundField {
                        name: "5".to_string(),
                        ty: <$t5 as H5Type>::type_descriptor(),
                        offset: mem::offset_of!(Self, 5),
                        index: 5,
                    },
                    CompoundField {
                        name: "6".to_string(),
                        ty: <$t6 as H5Type>::type_descriptor(),
                        offset: mem::offset_of!(Self, 6),
                        index: 6,
                    },
                    CompoundField {
                        name: "7".to_string(),
                        ty: <$t7 as H5Type>::type_descriptor(),
                        offset: mem::offset_of!(Self, 7),
                        index: 7,
                    },
                    CompoundField {
                        name: "8".to_string(),
                        ty: <$t8 as H5Type>::type_descriptor(),
                        offset: mem::offset_of!(Self, 8),
                        index: 8,
                    },
                    CompoundField {
                        name: "9".to_string(),
                        ty: <$t9 as H5Type>::type_descriptor(),
                        offset: mem::offset_of!(Self, 9),
                        index: 9,
                    },
                    CompoundField {
                        name: "10".to_string(),
                        ty: <$t10 as H5Type>::type_descriptor(),
                        offset: mem::offset_of!(Self, 10),
                        index: 10,
                    },
                ];
                let size = mem::size_of::<Self>();
                fields.sort_by_key(|f| f.offset);
                TypeDescriptor::Compound(CompoundType { fields, size })
            }
        }
    };
    // 12-tuple
    ([$t0:ident, $t1:ident, $t2:ident, $t3:ident, $t4:ident, $t5:ident, $t6:ident, $t7:ident, $t8:ident, $t9:ident, $t10:ident, $t11:ident] $($_idx:tt)*) => {
        #[allow(dead_code, unused_variables)]
        unsafe impl<$t0, $t1, $t2, $t3, $t4, $t5, $t6, $t7, $t8, $t9, $t10, $t11> H5Type for ($t0, $t1, $t2, $t3, $t4, $t5, $t6, $t7, $t8, $t9, $t10, $t11)
            where $t0: H5Type, $t1: H5Type, $t2: H5Type, $t3: H5Type, $t4: H5Type, $t5: H5Type, $t6: H5Type, $t7: H5Type, $t8: H5Type, $t9: H5Type, $t10: H5Type, $t11: H5Type
        {
            fn type_descriptor() -> TypeDescriptor {
                let mut fields = vec![
                    CompoundField {
                        name: "0".to_string(),
                        ty: <$t0 as H5Type>::type_descriptor(),
                        offset: mem::offset_of!(Self, 0),
                        index: 0,
                    },
                    CompoundField {
                        name: "1".to_string(),
                        ty: <$t1 as H5Type>::type_descriptor(),
                        offset: mem::offset_of!(Self, 1),
                        index: 1,
                    },
                    CompoundField {
                        name: "2".to_string(),
                        ty: <$t2 as H5Type>::type_descriptor(),
                        offset: mem::offset_of!(Self, 2),
                        index: 2,
                    },
                    CompoundField {
                        name: "3".to_string(),
                        ty: <$t3 as H5Type>::type_descriptor(),
                        offset: mem::offset_of!(Self, 3),
                        index: 3,
                    },
                    CompoundField {
                        name: "4".to_string(),
                        ty: <$t4 as H5Type>::type_descriptor(),
                        offset: mem::offset_of!(Self, 4),
                        index: 4,
                    },
                    CompoundField {
                        name: "5".to_string(),
                        ty: <$t5 as H5Type>::type_descriptor(),
                        offset: mem::offset_of!(Self, 5),
                        index: 5,
                    },
                    CompoundField {
                        name: "6".to_string(),
                        ty: <$t6 as H5Type>::type_descriptor(),
                        offset: mem::offset_of!(Self, 6),
                        index: 6,
                    },
                    CompoundField {
                        name: "7".to_string(),
                        ty: <$t7 as H5Type>::type_descriptor(),
                        offset: mem::offset_of!(Self, 7),
                        index: 7,
                    },
                    CompoundField {
                        name: "8".to_string(),
                        ty: <$t8 as H5Type>::type_descriptor(),
                        offset: mem::offset_of!(Self, 8),
                        index: 8,
                    },
                    CompoundField {
                        name: "9".to_string(),
                        ty: <$t9 as H5Type>::type_descriptor(),
                        offset: mem::offset_of!(Self, 9),
                        index: 9,
                    },
                    CompoundField {
                        name: "10".to_string(),
                        ty: <$t10 as H5Type>::type_descriptor(),
                        offset: mem::offset_of!(Self, 10),
                        index: 10,
                    },
                    CompoundField {
                        name: "11".to_string(),
                        ty: <$t11 as H5Type>::type_descriptor(),
                        offset: mem::offset_of!(Self, 11),
                        index: 11,
                    },
                ];
                let size = mem::size_of::<Self>();
                fields.sort_by_key(|f| f.offset);
                TypeDescriptor::Compound(CompoundType { fields, size })
            }
        }
    };
}

impl_tuple! { A, B, C, D, E, F, G, H, I, J, K, L }

unsafe impl<T: H5Type, const N: usize> H5Type for [T; N] {
    #[inline]
    fn type_descriptor() -> TypeDescriptor {
        TypeDescriptor::FixedArray(Box::new(<T as H5Type>::type_descriptor()), N)
    }
}

unsafe impl<T: Copy + H5Type> H5Type for VarLenArray<T> {
    #[inline]
    fn type_descriptor() -> TypeDescriptor {
        TypeDescriptor::VarLenArray(Box::new(<T as H5Type>::type_descriptor()))
    }
}

unsafe impl<const N: usize> H5Type for FixedAscii<N> {
    #[inline]
    fn type_descriptor() -> TypeDescriptor {
        TypeDescriptor::FixedAscii(N)
    }
}

unsafe impl<const N: usize> H5Type for FixedUnicode<N> {
    #[inline]
    fn type_descriptor() -> TypeDescriptor {
        TypeDescriptor::FixedUnicode(N)
    }
}

unsafe impl H5Type for VarLenAscii {
    #[inline]
    fn type_descriptor() -> TypeDescriptor {
        TypeDescriptor::VarLenAscii
    }
}

unsafe impl H5Type for VarLenUnicode {
    #[inline]
    fn type_descriptor() -> TypeDescriptor {
        TypeDescriptor::VarLenUnicode
    }
}

#[cfg(test)]
pub mod tests {
    use super::TypeDescriptor as TD;
    use super::{hvl_t, CompoundField, CompoundType, FloatSize, H5Type, IntSize};
    use crate::array::VarLenArray;
    use crate::string::{FixedAscii, FixedUnicode, VarLenAscii, VarLenUnicode};
    use std::mem;

    #[test]
    pub fn test_scalar_types() {
        assert_eq!(bool::type_descriptor(), TD::Boolean);
        assert_eq!(i8::type_descriptor(), TD::Integer(IntSize::U1));
        assert_eq!(i16::type_descriptor(), TD::Integer(IntSize::U2));
        assert_eq!(i32::type_descriptor(), TD::Integer(IntSize::U4));
        assert_eq!(i64::type_descriptor(), TD::Integer(IntSize::U8));
        assert_eq!(u8::type_descriptor(), TD::Unsigned(IntSize::U1));
        assert_eq!(u16::type_descriptor(), TD::Unsigned(IntSize::U2));
        assert_eq!(u32::type_descriptor(), TD::Unsigned(IntSize::U4));
        assert_eq!(u64::type_descriptor(), TD::Unsigned(IntSize::U8));
        assert_eq!(f32::type_descriptor(), TD::Float(FloatSize::U4));
        assert_eq!(f64::type_descriptor(), TD::Float(FloatSize::U8));

        assert_eq!(bool::type_descriptor().size(), 1);
        assert_eq!(i16::type_descriptor().size(), 2);
        assert_eq!(u32::type_descriptor().size(), 4);
        assert_eq!(f64::type_descriptor().size(), 8);
    }

    #[test]
    #[cfg(target_pointer_width = "32")]
    pub fn test_ptr_sized_ints() {
        assert_eq!(isize::type_descriptor(), TD::Integer(IntSize::U4));
        assert_eq!(usize::type_descriptor(), TD::Unsigned(IntSize::U4));

        assert_eq!(usize::type_descriptor().size(), 4);
    }

    #[test]
    #[cfg(target_pointer_width = "64")]
    pub fn test_ptr_sized_ints() {
        assert_eq!(isize::type_descriptor(), TD::Integer(IntSize::U8));
        assert_eq!(usize::type_descriptor(), TD::Unsigned(IntSize::U8));

        assert_eq!(usize::type_descriptor().size(), 8);
    }

    #[test]
    pub fn test_fixed_array() {
        type S = [T; 4];
        type T = [u32; 256];
        assert_eq!(T::type_descriptor(), TD::FixedArray(Box::new(TD::Unsigned(IntSize::U4)), 256));
        assert_eq!(S::type_descriptor(), TD::FixedArray(Box::new(T::type_descriptor()), 4));
    }

    #[test]
    pub fn test_varlen_array() {
        type S = VarLenArray<u16>;
        assert_eq!(S::type_descriptor(), TD::VarLenArray(Box::new(u16::type_descriptor())));
        assert_eq!(mem::size_of::<VarLenArray<u8>>(), mem::size_of::<hvl_t>());
    }

    #[test]
    pub fn test_string_types() {
        type FA = FixedAscii<16>;
        type FU = FixedUnicode<32>;
        assert_eq!(FA::type_descriptor(), TD::FixedAscii(16));
        assert_eq!(FU::type_descriptor(), TD::FixedUnicode(32));
        assert_eq!(VarLenAscii::type_descriptor(), TD::VarLenAscii);
        assert_eq!(VarLenUnicode::type_descriptor(), TD::VarLenUnicode);
    }

    #[test]
    pub fn test_tuples() {
        type T1 = (u16,);
        let td = T1::type_descriptor();
        assert_eq!(
            td,
            TD::Compound(CompoundType {
                fields: vec![CompoundField::typed::<u16>("0", 0, 0),],
                size: 2,
            })
        );
        assert_eq!(td.size(), 2);
        assert_eq!(mem::size_of::<T1>(), 2);

        type T2 = (i32, f32, (u64,));
        let td = T2::type_descriptor();
        assert_eq!(
            td,
            TD::Compound(CompoundType {
                fields: vec![
                    CompoundField::typed::<i32>("0", 0, 0),
                    CompoundField::typed::<f32>("1", 4, 1),
                    CompoundField::new(
                        "2",
                        TD::Compound(CompoundType {
                            fields: vec![CompoundField::typed::<u64>("0", 0, 0),],
                            size: 8,
                        }),
                        8,
                        2
                    ),
                ],
                size: 16,
            })
        );
        assert_eq!(td.size(), 16);
        assert_eq!(mem::size_of::<T2>(), 16);
    }

    #[test]
    pub fn test_tuple_various_reprs() {
        type T = (i8, u64, f32, bool);
        assert_eq!(mem::size_of::<T>(), 16);

        let td = T::type_descriptor();
        assert_eq!(
            td,
            TD::Compound(CompoundType {
                fields: vec![
                    CompoundField::typed::<u64>("1", 0, 1),
                    CompoundField::typed::<f32>("2", 8, 2),
                    CompoundField::typed::<i8>("0", 12, 0),
                    CompoundField::typed::<bool>("3", 13, 3),
                ],
                size: 16,
            })
        );
        assert_eq!(td.size(), 16);

        let td = T::type_descriptor().to_c_repr();
        assert_eq!(
            td,
            TD::Compound(CompoundType {
                fields: vec![
                    CompoundField::typed::<i8>("0", 0, 0),
                    CompoundField::typed::<u64>("1", 8, 1),
                    CompoundField::typed::<f32>("2", 16, 2),
                    CompoundField::typed::<bool>("3", 20, 3),
                ],
                size: 24,
            })
        );
        assert_eq!(td.size(), 24);

        let td = T::type_descriptor().to_packed_repr();
        assert_eq!(
            td,
            TD::Compound(CompoundType {
                fields: vec![
                    CompoundField::typed::<i8>("0", 0, 0),
                    CompoundField::typed::<u64>("1", 1, 1),
                    CompoundField::typed::<f32>("2", 9, 2),
                    CompoundField::typed::<bool>("3", 13, 3),
                ],
                size: 14,
            })
        );
        assert_eq!(td.size(), 14);
    }

    #[test]
    pub fn test_intsize_from_int_valid() {
        assert_eq!(IntSize::from_int(1), Some(IntSize::U1));
        assert_eq!(IntSize::from_int(2), Some(IntSize::U2));
        assert_eq!(IntSize::from_int(4), Some(IntSize::U4));
        assert_eq!(IntSize::from_int(8), Some(IntSize::U8));
    }

    #[test]
    pub fn test_intsize_from_int_invalid() {
        assert_eq!(IntSize::from_int(0), None);
        assert_eq!(IntSize::from_int(3), None);
        assert_eq!(IntSize::from_int(5), None);
        assert_eq!(IntSize::from_int(7), None);
        assert_eq!(IntSize::from_int(9), None);
        assert_eq!(IntSize::from_int(16), None);
        assert_eq!(IntSize::from_int(255), None);
    }

    #[test]
    pub fn test_floatsize_from_int_valid() {
        assert_eq!(FloatSize::from_int(4), Some(FloatSize::U4));
        assert_eq!(FloatSize::from_int(8), Some(FloatSize::U8));
        #[cfg(feature = "f16")]
        assert_eq!(FloatSize::from_int(2), Some(FloatSize::U2));
    }

    #[test]
    pub fn test_floatsize_from_int_invalid() {
        assert_eq!(FloatSize::from_int(0), None);
        assert_eq!(FloatSize::from_int(1), None);
        assert_eq!(FloatSize::from_int(3), None);
        assert_eq!(FloatSize::from_int(5), None);
        assert_eq!(FloatSize::from_int(16), None);
    }

    #[test]
    pub fn test_intsize_ord() {
        assert!(IntSize::U1 < IntSize::U2);
        assert!(IntSize::U2 < IntSize::U4);
        assert!(IntSize::U4 < IntSize::U8);
        assert_eq!(IntSize::U1, IntSize::U1);
    }

    #[test]
    pub fn test_floatsize_ord() {
        assert!(FloatSize::U4 < FloatSize::U8);
        assert_eq!(FloatSize::U4, FloatSize::U4);
    }

    #[test]
    pub fn test_enum_member() {
        let member = CompoundField::new("test", TD::Unsigned(IntSize::U4), 0, 0);
        assert_eq!(member.name, "test");
        assert_eq!(member.ty, TD::Unsigned(IntSize::U4));
        assert_eq!(member.offset, 0);
        assert_eq!(member.index, 0);
    }

    #[test]
    pub fn test_enum_type_base_type() {
        use super::EnumType;
        let enum_type = EnumType {
            size: IntSize::U4,
            signed: false,
            members: vec![],
        };
        assert_eq!(enum_type.base_type(), TD::Unsigned(IntSize::U4));

        let signed_enum = EnumType {
            size: IntSize::U2,
            signed: true,
            members: vec![],
        };
        assert_eq!(signed_enum.base_type(), TD::Integer(IntSize::U2));
    }

    #[test]
    pub fn test_compound_type_new() {
        let field1 = CompoundField::new("a", TD::Integer(IntSize::U4), 0, 0);
        let field2 = CompoundField::new("b", TD::Float(FloatSize::U8), 4, 1);
        let compound = CompoundType {
            fields: vec![field1, field2],
            size: 12,
        };
        assert_eq!(compound.fields.len(), 2);
        assert_eq!(compound.size, 12);
    }

    #[test]
    pub fn test_compound_type_to_c_repr_single_field() {
        let field = CompoundField::new("x", TD::Integer(IntSize::U4), 0, 0);
        let compound = CompoundType { fields: vec![field], size: 4 };
        let c_repr = compound.to_c_repr();
        assert_eq!(c_repr.size, 4);
        assert_eq!(c_repr.fields[0].offset, 0);
    }

    #[test]
    pub fn test_compound_type_to_packed_repr() {
        let field1 = CompoundField::typed::<u8>("a", 0, 0);
        let field2 = CompoundField::typed::<u64>("b", 8, 1);
        let compound = CompoundType { fields: vec![field1, field2], size: 16 };
        let packed = compound.to_packed_repr();
        assert_eq!(packed.size, 9); // 1 + 8, no padding
        assert_eq!(packed.fields[0].offset, 0);
        assert_eq!(packed.fields[1].offset, 1);
    }

    #[test]
    pub fn test_type_descriptor_size_fixed_array() {
        let td = TD::FixedArray(Box::new(TD::Integer(IntSize::U4)), 10);
        assert_eq!(td.size(), 40);
    }

    #[test]
    pub fn test_type_descriptor_size_varlen_array() {
        let td = TD::VarLenArray(Box::new(TD::Integer(IntSize::U4)));
        assert_eq!(td.size(), mem::size_of::<hvl_t>());
    }

    #[test]
    pub fn test_type_descriptor_size_fixed_string() {
        let td = TD::FixedAscii(32);
        assert_eq!(td.size(), 32);
        let td = TD::FixedUnicode(64);
        assert_eq!(td.size(), 64);
    }

    #[test]
    pub fn test_type_descriptor_size_varlen_string() {
        let td = TD::VarLenAscii;
        assert_eq!(td.size(), mem::size_of::<*const u8>());
        let td = TD::VarLenUnicode;
        assert_eq!(td.size(), mem::size_of::<*const u8>());
    }

    #[test]
    pub fn test_type_descriptor_c_alignment_primitive() {
        assert_eq!(TD::Integer(IntSize::U1).c_alignment(), 1);
        assert_eq!(TD::Integer(IntSize::U2).c_alignment(), 2);
        assert_eq!(TD::Integer(IntSize::U4).c_alignment(), 4);
        assert_eq!(TD::Integer(IntSize::U8).c_alignment(), 8);
        assert_eq!(TD::Unsigned(IntSize::U4).c_alignment(), 4);
        assert_eq!(TD::Float(FloatSize::U8).c_alignment(), 8);
        assert_eq!(TD::Boolean.c_alignment(), 1);
    }

    #[test]
    pub fn test_type_descriptor_c_alignment_array() {
        let td = TD::FixedArray(Box::new(TD::Integer(IntSize::U4)), 10);
        assert_eq!(td.c_alignment(), 4);
        let td = TD::FixedArray(Box::new(TD::Integer(IntSize::U8)), 5);
        assert_eq!(td.c_alignment(), 8);
    }

    #[test]
    pub fn test_type_descriptor_c_alignment_compound() {
        let field1 = CompoundField::typed::<u8>("a", 0, 0);
        let field2 = CompoundField::typed::<u64>("b", 8, 1);
        let compound = CompoundType { fields: vec![field1, field2], size: 16 };
        let td = TD::Compound(compound);
        assert_eq!(td.c_alignment(), 8); // max alignment
    }

    #[test]
    pub fn test_type_descriptor_to_c_repr_preserves_primitives() {
        assert_eq!(TD::Integer(IntSize::U4).to_c_repr(), TD::Integer(IntSize::U4));
        assert_eq!(TD::Float(FloatSize::U8).to_c_repr(), TD::Float(FloatSize::U8));
        assert_eq!(TD::Boolean.to_c_repr(), TD::Boolean);
    }

    #[test]
    pub fn test_type_descriptor_to_packed_repr_preserves_primitives() {
        assert_eq!(TD::Integer(IntSize::U4).to_packed_repr(), TD::Integer(IntSize::U4));
        assert_eq!(TD::Float(FloatSize::U8).to_packed_repr(), TD::Float(FloatSize::U8));
    }

    #[test]
    pub fn test_type_descriptor_display_integer() {
        assert_eq!(format!("{}", TD::Integer(IntSize::U1)), "int8");
        assert_eq!(format!("{}", TD::Integer(IntSize::U2)), "int16");
        assert_eq!(format!("{}", TD::Integer(IntSize::U4)), "int32");
        assert_eq!(format!("{}", TD::Integer(IntSize::U8)), "int64");
    }

    #[test]
    pub fn test_type_descriptor_display_unsigned() {
        assert_eq!(format!("{}", TD::Unsigned(IntSize::U1)), "uint8");
        assert_eq!(format!("{}", TD::Unsigned(IntSize::U2)), "uint16");
        assert_eq!(format!("{}", TD::Unsigned(IntSize::U4)), "uint32");
        assert_eq!(format!("{}", TD::Unsigned(IntSize::U8)), "uint64");
    }

    #[test]
    pub fn test_type_descriptor_display_float() {
        assert_eq!(format!("{}", TD::Float(FloatSize::U4)), "float32");
        assert_eq!(format!("{}", TD::Float(FloatSize::U8)), "float64");
    }

    #[test]
    pub fn test_type_descriptor_display_bool() {
        assert_eq!(format!("{}", TD::Boolean), "bool");
    }

    #[test]
    pub fn test_type_descriptor_display_enum() {
        use super::EnumType;
        let enum_type = EnumType {
            size: IntSize::U4,
            signed: false,
            members: vec![],
        };
        assert_eq!(format!("{}", TD::Enum(enum_type)), "enum (uint32)");
    }

    #[test]
    pub fn test_type_descriptor_display_compound() {
        let compound = CompoundType {
            fields: vec![CompoundField::typed::<u32>("x", 0, 0)],
            size: 4,
        };
        assert_eq!(format!("{}", TD::Compound(compound)), "compound (1 fields)");
    }

    #[test]
    pub fn test_type_descriptor_display_fixed_array() {
        let td = TD::FixedArray(Box::new(TD::Integer(IntSize::U4)), 10);
        assert_eq!(format!("{}", td), "[int32; 10]");
    }

    #[test]
    pub fn test_type_descriptor_display_varlen_array() {
        let td = TD::VarLenArray(Box::new(TD::Integer(IntSize::U4)));
        assert_eq!(format!("{}", td), "[int32] (var len)");
    }

    #[test]
    pub fn test_type_descriptor_display_strings() {
        assert_eq!(format!("{}", TD::FixedAscii(32)), "string (len 32)");
        assert_eq!(format!("{}", TD::FixedUnicode(64)), "unicode (len 64)");
        assert_eq!(format!("{}", TD::VarLenAscii), "string (var len)");
        assert_eq!(format!("{}", TD::VarLenUnicode), "unicode (var len)");
    }

    #[test]
    pub fn test_tuple_3_elements() {
        type T = (i32, f64, bool);
        let td = T::type_descriptor();
        assert!(matches!(td, TD::Compound(_)));
        assert_eq!(td.size(), mem::size_of::<T>());
    }

    #[test]
    pub fn test_tuple_5_elements() {
        type T = (u8, u16, u32, u64, i32);
        let td = T::type_descriptor();
        assert!(matches!(td, TD::Compound(_)));
        assert_eq!(td.size(), mem::size_of::<T>());
    }

    #[test]
    pub fn test_nested_tuple_compound() {
        type Inner = (u8, u16);
        type Outer = (i32, Inner, f64);
        let td = Outer::type_descriptor();
        assert!(matches!(td, TD::Compound(_)));
        assert_eq!(td.size(), mem::size_of::<Outer>());
    }

    #[test]
    pub fn test_compound_field_typed() {
        let field = CompoundField::typed::<u32>("my_field", 8, 1);
        assert_eq!(field.name, "my_field");
        assert_eq!(field.ty, TD::Unsigned(IntSize::U4));
        assert_eq!(field.offset, 8);
        assert_eq!(field.index, 1);
    }

    #[test]
    pub fn test_compound_type_with_multiple_fields_to_c_repr() {
        let fields = vec![
            CompoundField::typed::<u8>("a", 0, 0),
            CompoundField::typed::<u32>("b", 4, 1),
            CompoundField::typed::<u16>("c", 8, 2),
        ];
        let compound = CompoundType { fields, size: 12 };
        let c_repr = compound.to_c_repr();
        // C repr should align fields properly
        assert_eq!(c_repr.fields[0].offset, 0);
        assert!(c_repr.fields[1].offset >= 4);
        assert!(c_repr.fields[2].offset >= c_repr.fields[1].offset + 4);
    }
}
