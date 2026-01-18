use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;

use parking_lot::ReentrantMutex;

thread_local! {
    pub static SILENCED: AtomicBool = AtomicBool::new(false);
}

static LIBRARY_INIT: OnceLock<()> = OnceLock::new();

fn init_library() {
    // No functions called here must try to create the LOCK,
    // as this could cause a deadlock in initialisation
    unsafe {
        // Ensure hdf5 does not invalidate handles which might
        // still be live on other threads on program exit
        ::hdf5_sys::h5::H5dont_atexit();
        ::hdf5_sys::h5::H5open();
        // Ignore errors on stdout
        crate::error::silence_errors_no_sync(true);
        // Register filters lzf/blosc if available
        crate::hl::filters::register_filters();
    }
}

pub(crate) fn ensure_library_init() {
    LIBRARY_INIT.get_or_init(|| init_library());
}

/// Guards the execution of the provided closure with a recursive static mutex.
pub fn sync<T, F>(func: F) -> T
where
    F: FnOnce() -> T,
{
    static LOCK: OnceLock<ReentrantMutex<()>> = OnceLock::new();

    ensure_library_init();

    let lock = LOCK.get_or_init(|| ReentrantMutex::new(()));
    SILENCED.with(|silence| {
        let is_silenced = silence.load(Ordering::Acquire);
        if !is_silenced {
            let _guard = lock.lock();
            unsafe {
                crate::error::silence_errors_no_sync(true);
            }
            silence.store(true, Ordering::Release);
        }
    });
    let _guard = lock.lock();
    func()
}

#[cfg(test)]
mod tests {
    use parking_lot::ReentrantMutex;
    use std::sync::OnceLock;

    #[test]
    pub fn test_reentrant_mutex() {
        static LOCK: OnceLock<ReentrantMutex<()>> = OnceLock::new();
        let lock = LOCK.get_or_init(|| ReentrantMutex::new(()));

        let g1 = lock.try_lock();
        assert!(g1.is_some());
        let g2 = lock.lock();
        assert_eq!(*g2, ());
        let g3 = lock.try_lock();
        assert!(g3.is_some());
        let g4 = lock.lock();
        assert_eq!(*g4, ());
    }

    #[test]
    // Test for locking behaviour on initialisation
    pub fn lock_part1() {
        let _ = *crate::globals::H5P_ROOT;
    }

    #[test]
    // Test for locking behaviour on initialisation
    pub fn lock_part2() {
        let _ = h5call!(*crate::globals::H5P_ROOT);
    }
}
