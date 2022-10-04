#pragma once 

#ifdef CUDAVEC
   #define CheckErrors(msg) \
      do { \
         cudaError_t __err = cudaGetLastError(); \
         if (__err != cudaSuccess) { \
               fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                  msg, cudaGetErrorString(__err), \
                  __FILE__, __LINE__); \
               fprintf(stderr, "***** FAILED - ABORTING*****\n"); \
               exit(1); \
         } \
      } while (0)
#else
//TODO--> make it do smth.
#pragma message ("TODO-->Make this a no NOOP" )
   #define CheckErrors(msg) \
      do { }  while (0)
#endif

namespace split{
   template <typename T>
   class split_host_allocator {
   public:
       // naming tradition
       typedef T value_type;
       typedef T *pointer;
       typedef const T *const_pointer;
       typedef T &reference;
       typedef const T &const_reference;
       typedef std::size_t size_type;
       typedef std::ptrdiff_t difference_type;

       template <typename U> struct rebind {typedef split_host_allocator<U> other;};
       split_host_allocator() = default;
       split_host_allocator(const split_host_allocator &) {}
       template <typename U> split_host_allocator(const split_host_allocator<U> &other) {}
       split_host_allocator &operator=(const split_host_allocator &) = delete;
       ~split_host_allocator() = default;
       //Members
       pointer address(reference r) {return &r;}
       const_pointer address(const_reference cr) {return &cr;}
       size_type max_size() {return std::numeric_limits<size_type>::max();}
       bool operator==(const split_host_allocator &) const {return true;}
       bool operator!=(const split_host_allocator &) const {return false;}
       pointer allocate(size_type n) {return static_cast<pointer>(operator new(sizeof(T) * n));}
       pointer allocate(size_type n, pointer ptr) {return allocate(n);}
       void deallocate(pointer ptr, size_type n) {operator delete(ptr);}
       void construct(pointer ptr, const value_type &t) {new(ptr) value_type(t);}
       void destroy(pointer ptr) {ptr->~value_type();}
       pointer allocate_and_construct(size_type n , const value_type &t){return new T[n]; }
       void deallocate_array(size_type n, pointer ptr){
          delete []ptr;
       }
       void* allocate_raw(size_t bytes){ return operator new(bytes);}
       template<typename  C>
       void deallocate_raw(C* ptr){delete ptr; }

   };


   template <typename T>
   class split_unified_allocator{
   public:
       // naming tradition
       typedef T value_type;
       typedef T *pointer;
       typedef const T *const_pointer;
       typedef T &reference;
       typedef const T &const_reference;
       typedef std::size_t size_type;
       typedef std::ptrdiff_t difference_type;

       template <typename U> struct rebind {typedef split_unified_allocator<U> other;};
       split_unified_allocator() = default;
       split_unified_allocator(const split_unified_allocator &) {}
       template <typename U> split_unified_allocator(const split_unified_allocator<U> &other) {}
       split_unified_allocator &operator=(const split_unified_allocator &) = delete;
       ~split_unified_allocator() = default;
       //Members
       pointer address(reference r) {return &r;}
       const_pointer address(const_reference cr) {return &cr;}
       size_type max_size() {return std::numeric_limits<size_type>::max();}
       bool operator==(const split_unified_allocator &) const {return true;}
       bool operator!=(const split_unified_allocator &) const {return false;}


       pointer allocate_and_construct(size_type n , const value_type &t){
          T* ptr;
          cudaMallocManaged((void**)&ptr, n * sizeof(T));
          for (size_t i=0; i < n; i++){
             new (&ptr[i]) T(); 
          }
          CheckErrors("Managed Allocation");
          return ptr;
       }

       void deallocate_array(size_type n, pointer ptr){
          for (size_type i=0; i<n;i++){
             ptr[i].~T();
          }
          cudaFree(ptr);
          CheckErrors("Managed Deallocation");
       }
       void* allocate_raw(size_t bytes){
          void *ptr;
          cudaMallocManaged((void**)&ptr,bytes);
          CheckErrors("Managed Allocation");
          return ptr;
       }
       template<typename  C>
       void deallocate_raw(C* ptr){
          cudaFree(ptr);
          CheckErrors("Managed Deallocation");
       }

   };
}
