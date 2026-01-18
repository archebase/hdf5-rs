use std::borrow::Borrow;
use std::cmp::{Ordering, PartialEq, PartialOrd};
use std::fmt::{self, Debug, Display};
use std::ops::Deref;
use std::ptr::{addr_of, addr_of_mut};

use hdf5_sys::h5t::{
    H5T_class_t, H5T_cset_t, H5T_order_t, H5T_sign_t, H5T_str_t, H5Tarray_create2,
    H5Tcompiler_conv, H5Tcopy, H5Tcreate, H5Tenum_create, H5Tenum_insert, H5Tequal,
    H5Tget_array_dims2, H5Tget_array_ndims, H5Tget_class, H5Tget_cset, H5Tget_member_name,
    H5Tget_member_offset, H5Tget_member_type, H5Tget_member_value, H5Tget_nmembers, H5Tget_order,
    H5Tget_sign, H5Tget_size, H5Tget_super, H5Tinsert, H5Tis_variable_str, H5Tset_cset,
    H5Tset_size, H5Tset_strpad, H5Tvlen_create, H5T_VARIABLE,
};
use hdf5_types::{
    CompoundField, CompoundType, EnumMember, EnumType, FloatSize, H5Type, IntSize, TypeDescriptor,
};

use crate::globals::{H5T_C_S1, H5T_NATIVE_INT8};
use crate::internal_prelude::*;

#[cfg(target_endian = "big")]
use crate::globals::{
    H5T_IEEE_F32BE, H5T_IEEE_F64BE, H5T_STD_I16BE, H5T_STD_I32BE, H5T_STD_I64BE, H5T_STD_I8BE,
    H5T_STD_U16BE, H5T_STD_U32BE, H5T_STD_U64BE, H5T_STD_U8BE,
};

#[cfg(target_endian = "little")]
use crate::globals::{
    H5T_IEEE_F32LE, H5T_IEEE_F64LE, H5T_STD_I16LE, H5T_STD_I32LE, H5T_STD_I64LE, H5T_STD_I8LE,
    H5T_STD_U16LE, H5T_STD_U32LE, H5T_STD_U64LE, H5T_STD_U8LE,
};

#[cfg(target_endian = "big")]
macro_rules! be_le {
    ($be:expr, $le:expr) => {
        h5try!(H5Tcopy(*$be))
    };
}

#[cfg(target_endian = "little")]
macro_rules! be_le {
    ($be:expr, $le:expr) => {
        h5try!(H5Tcopy(*$le))
    };
}

/// Represents the HDF5 datatype object.
#[repr(transparent)]
#[derive(Clone)]
pub struct Datatype(Handle);

impl ObjectClass for Datatype {
    const NAME: &'static str = "datatype";
    const VALID_TYPES: &'static [H5I_type_t] = &[H5I_DATATYPE];

    fn from_handle(handle: Handle) -> Self {
        Self(handle)
    }

    fn handle(&self) -> &Handle {
        &self.0
    }

    fn short_repr(&self) -> Option<String> {
        Some(format!("<datatype id={}>", self.id()))
    }
}

impl Debug for Datatype {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.debug_fmt(f)
    }
}

impl Deref for Datatype {
    type Target = Object;

    fn deref(&self) -> &Object {
        unsafe { self.transmute() }
    }
}

impl PartialEq for Datatype {
    fn eq(&self, other: &Self) -> bool {
        h5call!(H5Tequal(self.id(), other.id())).unwrap_or(0) == 1
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Conversion {
    NoOp = 1, // TODO: rename to "None"?
    Hard,
    Soft,
}

impl PartialEq<Conversion> for Option<Conversion> {
    fn eq(&self, _other: &Conversion) -> bool {
        false
    }
}

impl PartialOrd<Conversion> for Option<Conversion> {
    fn partial_cmp(&self, other: &Conversion) -> Option<Ordering> {
        self.map(|conv| conv.partial_cmp(other)).unwrap_or(Some(Ordering::Greater))
    }
}

impl Display for Conversion {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(match self {
            Self::NoOp => "no-op",
            Self::Hard => "hard",
            Self::Soft => "soft",
        })
    }
}

impl Default for Conversion {
    fn default() -> Self {
        Self::NoOp
    }
}

#[derive(Copy, Debug, Clone, PartialEq, Eq)]
pub enum ByteOrder {
    LittleEndian,
    BigEndian,
    Vax,
    Mixed,
    None,
}

#[cfg(feature = "1.8.6")]
impl From<H5T_order_t> for ByteOrder {
    fn from(order: H5T_order_t) -> Self {
        match order {
            H5T_order_t::H5T_ORDER_LE => Self::LittleEndian,
            H5T_order_t::H5T_ORDER_BE => Self::BigEndian,
            H5T_order_t::H5T_ORDER_VAX => Self::Vax,
            H5T_order_t::H5T_ORDER_MIXED => Self::Mixed,
            _ => Self::None,
        }
    }
}

#[cfg(not(feature = "1.8.6"))]
impl From<H5T_order_t> for ByteOrder {
    fn from(order: H5T_order_t) -> Self {
        match order {
            H5T_order_t::H5T_ORDER_LE => Self::LittleEndian,
            H5T_order_t::H5T_ORDER_BE => Self::BigEndian,
            H5T_order_t::H5T_ORDER_VAX => Self::Vax,
            _ => Self::None,
        }
    }
}

impl Datatype {
    /// Get the total size of the datatype in bytes.
    #[allow(clippy::unnecessary_cast)]
    pub fn size(&self) -> usize {
        h5lock!(H5Tget_size(self.id())) as usize
    }

    /// Get the byte order of the datatype.
    pub fn byte_order(&self) -> ByteOrder {
        h5lock!(H5Tget_order(self.id())).into()
    }

    pub fn conv_path<D>(&self, dst: D) -> Option<Conversion>
    where
        D: Borrow<Self>,
    {
        let dst = dst.borrow();
        h5lock!({
            // Check for no-op conversion by comparing type IDs directly
            // This is more reliable than comparing function pointers returned by H5Tfind
            if self.id() == dst.id() || H5Tequal(self.id(), dst.id()) > 0 {
                Some(Conversion::NoOp)
            } else {
                match H5Tcompiler_conv(self.id(), dst.id()) {
                    0 => Some(Conversion::Soft),
                    r if r > 0 => Some(Conversion::Hard),
                    _ => None,
                }
            }
        })
    }

    pub fn conv_to<T: H5Type>(&self) -> Option<Conversion> {
        Self::from_type::<T>().ok().and_then(|dtype| self.conv_path(dtype))
    }

    pub fn conv_from<T: H5Type>(&self) -> Option<Conversion> {
        Self::from_type::<T>().ok().and_then(|dtype| dtype.conv_path(self))
    }

    pub fn is<T: H5Type>(&self) -> bool {
        Self::from_type::<T>().ok().map_or(false, |dtype| &dtype == self)
    }

    pub(crate) fn ensure_convertible(&self, dst: &Self, required: Conversion) -> Result<()> {
        // TODO: more detailed error messages after Debug/Display are implemented for Datatype
        if let Some(conv) = self.conv_path(dst) {
            ensure!(
                conv <= required,
                "{} conversion path required; available: {} conversion",
                required,
                conv
            );
            Ok(())
        } else {
            fail!("no conversion paths found")
        }
    }

    pub fn to_descriptor(&self) -> Result<TypeDescriptor> {
        use hdf5_types::TypeDescriptor as TD;

        h5lock!({
            let id = self.id();
            #[allow(clippy::unnecessary_cast)]
            let size = H5Tget_size(id) as usize;
            match H5Tget_class(id) {
                H5T_class_t::H5T_INTEGER => {
                    let signed = match H5Tget_sign(id) {
                        H5T_sign_t::H5T_SGN_NONE => false,
                        H5T_sign_t::H5T_SGN_2 => true,
                        _ => return Err("Invalid sign of integer datatype".into()),
                    };
                    let size = IntSize::from_int(size).ok_or("Invalid size of integer datatype")?;
                    Ok(if signed { TD::Integer(size) } else { TD::Unsigned(size) })
                }
                H5T_class_t::H5T_FLOAT => {
                    let size = FloatSize::from_int(size).ok_or("Invalid size of float datatype")?;
                    Ok(TD::Float(size))
                }
                H5T_class_t::H5T_ENUM => {
                    let mut members: Vec<EnumMember> = Vec::new();
                    for idx in 0..h5try!(H5Tget_nmembers(id)) as _ {
                        let mut value: u64 = 0;
                        h5try!(H5Tget_member_value(id, idx, addr_of_mut!(value).cast()));
                        let name = H5Tget_member_name(id, idx);
                        members.push(EnumMember { name: string_from_cstr(name), value });
                        h5_free_memory(name.cast());
                    }
                    let base_dt = Self::from_id(H5Tget_super(id))?;
                    let (size, signed) = match base_dt.to_descriptor()? {
                        TD::Integer(size) => Ok((size, true)),
                        TD::Unsigned(size) => Ok((size, false)),
                        _ => Err("Invalid base type for enum datatype"),
                    }?;
                    let bool_members = [
                        EnumMember { name: "FALSE".to_owned(), value: 0 },
                        EnumMember { name: "TRUE".to_owned(), value: 1 },
                    ];
                    if size == IntSize::U1 && members == bool_members {
                        Ok(TD::Boolean)
                    } else {
                        Ok(TD::Enum(EnumType { size, signed, members }))
                    }
                }
                H5T_class_t::H5T_COMPOUND => {
                    let mut fields: Vec<CompoundField> = Vec::new();
                    for idx in 0..h5try!(H5Tget_nmembers(id)) as _ {
                        let name = H5Tget_member_name(id, idx);
                        let offset = H5Tget_member_offset(id, idx);
                        let ty = Self::from_id(h5try!(H5Tget_member_type(id, idx)))?;
                        fields.push(CompoundField {
                            name: string_from_cstr(name),
                            ty: ty.to_descriptor()?,
                            offset: offset as _,
                            index: idx as _,
                        });
                        h5_free_memory(name.cast());
                    }
                    Ok(TD::Compound(CompoundType { fields, size }))
                }
                H5T_class_t::H5T_ARRAY => {
                    let base_dt = Self::from_id(H5Tget_super(id))?;
                    let ndims = h5try!(H5Tget_array_ndims(id));
                    if ndims == 1 {
                        let mut len: hsize_t = 0;
                        h5try!(H5Tget_array_dims2(id, addr_of_mut!(len)));
                        Ok(TD::FixedArray(Box::new(base_dt.to_descriptor()?), len as _))
                    } else {
                        Err("Multi-dimensional array datatypes are not supported".into())
                    }
                }
                H5T_class_t::H5T_STRING => {
                    let is_variable = h5try!(H5Tis_variable_str(id)) == 1;
                    let encoding = h5lock!(H5Tget_cset(id));
                    match (is_variable, encoding) {
                        (false, H5T_cset_t::H5T_CSET_ASCII) => Ok(TD::FixedAscii(size)),
                        (false, H5T_cset_t::H5T_CSET_UTF8) => Ok(TD::FixedUnicode(size)),
                        (true, H5T_cset_t::H5T_CSET_ASCII) => Ok(TD::VarLenAscii),
                        (true, H5T_cset_t::H5T_CSET_UTF8) => Ok(TD::VarLenUnicode),
                        _ => Err("Invalid encoding for string datatype".into()),
                    }
                }
                H5T_class_t::H5T_VLEN => {
                    let base_dt = Self::from_id(H5Tget_super(id))?;
                    Ok(TD::VarLenArray(Box::new(base_dt.to_descriptor()?)))
                }
                _ => Err("Unsupported datatype class".into()),
            }
        })
    }

    pub fn from_type<T: H5Type>() -> Result<Self> {
        Self::from_descriptor(&<T as H5Type>::type_descriptor())
    }

    pub fn from_descriptor(desc: &TypeDescriptor) -> Result<Self> {
        use hdf5_types::TypeDescriptor as TD;

        unsafe fn string_type(size: Option<usize>, encoding: H5T_cset_t) -> Result<hid_t> {
            let string_id = h5try!(H5Tcopy(*H5T_C_S1));
            let padding = if size.is_none() {
                H5T_str_t::H5T_STR_NULLTERM
            } else {
                H5T_str_t::H5T_STR_NULLPAD
            };
            let size = size.unwrap_or(H5T_VARIABLE);
            h5try!(H5Tset_cset(string_id, encoding));
            h5try!(H5Tset_strpad(string_id, padding));
            h5try!(H5Tset_size(string_id, size));
            Ok(string_id)
        }

        #[cfg(feature = "f16")]
        unsafe fn f16_type() -> Result<hid_t> {
            use hdf5_sys::h5t::{H5Tset_ebias, H5Tset_fields};
            let f16_id = be_le!(H5T_IEEE_F32BE, H5T_IEEE_F32LE);
            h5try!(H5Tset_fields(f16_id, 15, 10, 5, 0, 10)); // cf. h5py/h5py#339
            h5try!(H5Tset_size(f16_id, 2));
            h5try!(H5Tset_ebias(f16_id, 15));
            Ok(f16_id)
        }

        let datatype_id: Result<_> = h5lock!({
            match *desc {
                TD::Integer(size) => Ok(match size {
                    IntSize::U1 => be_le!(H5T_STD_I8BE, H5T_STD_I8LE),
                    IntSize::U2 => be_le!(H5T_STD_I16BE, H5T_STD_I16LE),
                    IntSize::U4 => be_le!(H5T_STD_I32BE, H5T_STD_I32LE),
                    IntSize::U8 => be_le!(H5T_STD_I64BE, H5T_STD_I64LE),
                }),
                TD::Unsigned(size) => Ok(match size {
                    IntSize::U1 => be_le!(H5T_STD_U8BE, H5T_STD_U8LE),
                    IntSize::U2 => be_le!(H5T_STD_U16BE, H5T_STD_U16LE),
                    IntSize::U4 => be_le!(H5T_STD_U32BE, H5T_STD_U32LE),
                    IntSize::U8 => be_le!(H5T_STD_U64BE, H5T_STD_U64LE),
                }),
                TD::Float(size) => Ok(match size {
                    #[cfg(feature = "f16")]
                    FloatSize::U2 => f16_type()?,
                    FloatSize::U4 => be_le!(H5T_IEEE_F32BE, H5T_IEEE_F32LE),
                    FloatSize::U8 => be_le!(H5T_IEEE_I16BE, H5T_IEEE_F64LE),
                }),
                TD::Boolean => {
                    let bool_id = h5try!(H5Tenum_create(*H5T_NATIVE_INT8));
                    let zero = 0_i8;
                    h5try!(H5Tenum_insert(
                        bool_id,
                        b"FALSE\0".as_ptr().cast(),
                        addr_of!(zero).cast(),
                    ));
                    let one = 1_i8;
                    h5try!(H5Tenum_insert(
                        bool_id,
                        b"TRUE\0".as_ptr().cast(),
                        addr_of!(one).cast(),
                    ));
                    Ok(bool_id)
                }
                TD::Enum(ref enum_type) => {
                    let base = Self::from_descriptor(&enum_type.base_type())?;
                    let enum_id = h5try!(H5Tenum_create(base.id()));
                    for member in &enum_type.members {
                        let name = to_cstring(member.name.as_ref())?;
                        h5try!(H5Tenum_insert(
                            enum_id,
                            name.as_ptr(),
                            addr_of!(member.value).cast()
                        ));
                    }
                    Ok(enum_id)
                }
                TD::Compound(ref compound_type) => {
                    let compound_id = h5try!(H5Tcreate(H5T_class_t::H5T_COMPOUND, 1));
                    for field in &compound_type.fields {
                        let name = to_cstring(field.name.as_ref())?;
                        let field_dt = Self::from_descriptor(&field.ty)?;
                        h5try!(H5Tset_size(compound_id, field.offset + field.ty.size()));
                        h5try!(H5Tinsert(compound_id, name.as_ptr(), field.offset, field_dt.id()));
                    }
                    h5try!(H5Tset_size(compound_id, compound_type.size));
                    Ok(compound_id)
                }
                TD::FixedArray(ref ty, len) => {
                    let elem_dt = Self::from_descriptor(ty)?;
                    let dims = len as hsize_t;
                    Ok(h5try!(H5Tarray_create2(elem_dt.id(), 1, addr_of!(dims))))
                }
                TD::FixedAscii(size) => string_type(Some(size), H5T_cset_t::H5T_CSET_ASCII),
                TD::FixedUnicode(size) => string_type(Some(size), H5T_cset_t::H5T_CSET_UTF8),
                TD::VarLenArray(ref ty) => {
                    let elem_dt = Self::from_descriptor(ty)?;
                    Ok(h5try!(H5Tvlen_create(elem_dt.id())))
                }
                TD::VarLenAscii => string_type(None, H5T_cset_t::H5T_CSET_ASCII),
                TD::VarLenUnicode => string_type(None, H5T_cset_t::H5T_CSET_UTF8),
            }
        });

        Self::from_id(datatype_id?)
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;

    #[test]
    pub fn test_conversion_display() {
        assert_eq!(format!("{}", Conversion::NoOp), "no-op");
        assert_eq!(format!("{}", Conversion::Soft), "soft");
        assert_eq!(format!("{}", Conversion::Hard), "hard");
    }

    #[test]
    pub fn test_conversion_default() {
        assert_eq!(Conversion::default(), Conversion::NoOp);
    }

    #[test]
    pub fn test_conversion_partial_ord() {
        assert!(Conversion::NoOp < Conversion::Hard);
        assert!(Conversion::Hard < Conversion::Soft);
        assert_eq!(Conversion::NoOp, Conversion::NoOp);
    }

    #[test]
    pub fn test_conversion_ord() {
        assert!(Conversion::NoOp < Conversion::Hard);
        assert!(Conversion::Hard < Conversion::Soft);
    }

    #[test]
    pub fn test_option_conversion_partial_eq() {
        // Option<Conversion> is never equal to Conversion
        assert_eq!(Option::<Conversion>::None == Conversion::NoOp, false);
        assert_eq!(Some(Conversion::NoOp) == Conversion::NoOp, false);
    }

    #[test]
    pub fn test_option_conversion_partial_cmp() {
        // Option<Conversion> partial cmp with Conversion
        assert_eq!(
            Option::<Conversion>::None.partial_cmp(&Conversion::NoOp),
            Some(Ordering::Greater)
        );
        // NoOp < Hard < Soft
        assert_eq!(Some(Conversion::NoOp).partial_cmp(&Conversion::Hard), Some(Ordering::Less));
        assert_eq!(Some(Conversion::Hard).partial_cmp(&Conversion::Soft), Some(Ordering::Less));
        // Greater cases
        assert_eq!(Some(Conversion::Soft).partial_cmp(&Conversion::Hard), Some(Ordering::Greater));
    }

    #[test]
    pub fn test_datatype_size_i32() {
        let dt = Datatype::from_type::<i32>().unwrap();
        assert_eq!(dt.size(), 4);
    }

    #[test]
    pub fn test_datatype_size_f64() {
        let dt = Datatype::from_type::<f64>().unwrap();
        assert_eq!(dt.size(), 8);
    }

    #[test]
    pub fn test_datatype_size_bool() {
        let dt = Datatype::from_type::<bool>().unwrap();
        assert_eq!(dt.size(), 1);
    }

    #[test]
    pub fn test_datatype_byte_order() {
        let dt = Datatype::from_type::<i32>().unwrap();
        let order = dt.byte_order();
        // Should be either LittleEndian or BigEndian depending on platform
        assert!(matches!(order, ByteOrder::LittleEndian | ByteOrder::BigEndian));
    }

    #[test]
    pub fn test_datatype_is_same_type() {
        let dt_i32 = Datatype::from_type::<i32>().unwrap();
        assert!(dt_i32.is::<i32>());
        assert!(!dt_i32.is::<f32>());
        assert!(!dt_i32.is::<i64>());
    }

    #[test]
    pub fn test_datatype_is_different_types() {
        let dt_f32 = Datatype::from_type::<f32>().unwrap();
        assert!(dt_f32.is::<f32>());
        assert!(!dt_f32.is::<i32>());
        assert!(!dt_f32.is::<f64>());
    }

    #[test]
    pub fn test_datatype_conv_path_noop() {
        let src = Datatype::from_type::<i32>().unwrap();
        let dst = Datatype::from_type::<i32>().unwrap();
        // Same type should be NoOp conversion
        assert_eq!(src.conv_path(&dst), Some(Conversion::NoOp));
        assert_eq!(src.conv_path(&src), Some(Conversion::NoOp));
    }

    #[test]
    pub fn test_datatype_conv_path_soft() {
        let src = Datatype::from_type::<i32>().unwrap();
        let dst = Datatype::from_type::<i64>().unwrap();
        // Integer to integer (same sign) should be Soft conversion
        let conv = src.conv_path(&dst);
        assert!(conv.is_some());
        // i32 to f32 would be Hard conversion
        let dt_f32 = Datatype::from_type::<f32>().unwrap();
        let conv2 = src.conv_path(&dt_f32);
        assert!(conv2.is_some());
    }

    #[test]
    pub fn test_datatype_conv_to() {
        let dt_i32 = Datatype::from_type::<i32>().unwrap();
        // Converting to same type
        assert_eq!(dt_i32.conv_to::<i32>(), Some(Conversion::NoOp));
        // Converting to different type
        assert!(dt_i32.conv_to::<i64>().is_some());
        assert!(dt_i32.conv_to::<f32>().is_some());
    }

    #[test]
    pub fn test_datatype_conv_from() {
        let dt_i32 = Datatype::from_type::<i32>().unwrap();
        // Converting from same type
        assert_eq!(dt_i32.conv_from::<i32>(), Some(Conversion::NoOp));
        // Converting from different type
        assert!(dt_i32.conv_from::<i64>().is_some());
        assert!(dt_i32.conv_from::<f32>().is_some());
    }

    #[test]
    pub fn test_datatype_equality() {
        let dt1 = Datatype::from_type::<i32>().unwrap();
        let dt2 = Datatype::from_type::<i32>().unwrap();
        assert_eq!(dt1, dt2);

        let dt_f32 = Datatype::from_type::<f32>().unwrap();
        assert_ne!(dt1, dt_f32);
    }

    #[test]
    pub fn test_datatype_to_descriptor_i32() {
        let dt = Datatype::from_type::<i32>().unwrap();
        let desc = dt.to_descriptor().unwrap();
        assert!(matches!(desc, hdf5_types::TypeDescriptor::Integer(_)));
    }

    #[test]
    pub fn test_datatype_to_descriptor_u32() {
        let dt = Datatype::from_type::<u32>().unwrap();
        let desc = dt.to_descriptor().unwrap();
        assert!(matches!(desc, hdf5_types::TypeDescriptor::Unsigned(_)));
    }

    #[test]
    pub fn test_datatype_to_descriptor_f64() {
        let dt = Datatype::from_type::<f64>().unwrap();
        let desc = dt.to_descriptor().unwrap();
        assert!(matches!(desc, hdf5_types::TypeDescriptor::Float(_)));
    }

    #[test]
    pub fn test_datatype_to_descriptor_bool() {
        let dt = Datatype::from_type::<bool>().unwrap();
        let desc = dt.to_descriptor().unwrap();
        assert_eq!(desc, hdf5_types::TypeDescriptor::Boolean);
    }

    #[test]
    pub fn test_datatype_from_descriptor_i32() {
        let desc = hdf5_types::TypeDescriptor::Integer(hdf5_types::IntSize::U4);
        let dt = Datatype::from_descriptor(&desc).unwrap();
        assert!(dt.is::<i32>());
    }

    #[test]
    pub fn test_datatype_from_descriptor_u32() {
        let desc = hdf5_types::TypeDescriptor::Unsigned(hdf5_types::IntSize::U4);
        let dt = Datatype::from_descriptor(&desc).unwrap();
        assert!(dt.is::<u32>());
    }

    #[test]
    pub fn test_datatype_from_descriptor_f32() {
        let desc = hdf5_types::TypeDescriptor::Float(hdf5_types::FloatSize::U4);
        let dt = Datatype::from_descriptor(&desc).unwrap();
        assert!(dt.is::<f32>());
    }

    #[test]
    pub fn test_datatype_from_descriptor_bool() {
        let desc = hdf5_types::TypeDescriptor::Boolean;
        let dt = Datatype::from_descriptor(&desc).unwrap();
        assert!(dt.is::<bool>());
    }

    #[test]
    pub fn test_datatype_roundtrip_i32() {
        let dt1 = Datatype::from_type::<i32>().unwrap();
        let desc = dt1.to_descriptor().unwrap();
        let dt2 = Datatype::from_descriptor(&desc).unwrap();
        assert_eq!(dt1, dt2);
        assert!(dt2.is::<i32>());
    }

    #[test]
    pub fn test_datatype_roundtrip_f64() {
        let dt1 = Datatype::from_type::<f64>().unwrap();
        let desc = dt1.to_descriptor().unwrap();
        let dt2 = Datatype::from_descriptor(&desc).unwrap();
        assert_eq!(dt1, dt2);
        assert!(dt2.is::<f64>());
    }

    #[test]
    pub fn test_datatype_roundtrip_bool() {
        let dt1 = Datatype::from_type::<bool>().unwrap();
        let desc = dt1.to_descriptor().unwrap();
        let dt2 = Datatype::from_descriptor(&desc).unwrap();
        assert_eq!(dt1, dt2);
        assert!(dt2.is::<bool>());
    }

    #[test]
    pub fn test_datatype_from_descriptor_array() {
        let elem_desc = hdf5_types::TypeDescriptor::Integer(hdf5_types::IntSize::U4);
        let desc = hdf5_types::TypeDescriptor::FixedArray(Box::new(elem_desc), 10);
        let dt = Datatype::from_descriptor(&desc).unwrap();
        assert_eq!(dt.size(), 40);
    }

    #[test]
    pub fn test_datatype_from_descriptor_fixed_string() {
        let desc = hdf5_types::TypeDescriptor::FixedAscii(32);
        let dt = Datatype::from_descriptor(&desc).unwrap();
        assert_eq!(dt.size(), 32);
    }

    #[test]
    pub fn test_datatype_from_descriptor_varlen_string() {
        let desc = hdf5_types::TypeDescriptor::VarLenAscii;
        let dt = Datatype::from_descriptor(&desc).unwrap();
        // VarLen strings have pointer size
        assert!(
            dt.size() == std::mem::size_of::<*const u8>()
                || dt.size() == std::mem::size_of::<usize>()
        );
    }

    #[test]
    pub fn test_datatype_debug() {
        let dt = Datatype::from_type::<i32>().unwrap();
        let debug_str = format!("{:?}", dt);
        assert!(debug_str.contains("datatype"));
    }
}
