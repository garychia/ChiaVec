#ifndef __CHIAVEC_RAWVECS_HPP__
#define __CHIAVEC_RAWVECS_HPP__

#include "allocators.hpp"

namespace ChiaVec
{
    namespace Raw
    {
        template <class T, class Allocator = Memory::DefaultAllocator>
        class RawVec
        {
        private:
            T *elements;
            size_t length;

            void releaseMemory()
            {
                Allocator allocator;
                allocator.release(elements);
            }

            void deleteElements()
            {
                releaseMemory();
                elements = nullptr;
            }

            void clearAll()
            {
                deleteElements();
                length = 0;
            }

            virtual void copyElement(const T &element, std::size_t index)
            {
                Allocator allocator;
                allocator.copy(&elements[index], &element, sizeof(T));
            }

        public:
            RawVec() : elements(nullptr), length(0)
            {
            }

            RawVec(std::size_t length) : length(length)
            {
                Allocator allocator;
                elements = static_cast<T *>(allocator(sizeof(T) * length));
            }

            RawVec(const T *data, std::size_t length) : RawVec(length)
            {
                Allocator allocator;
                allocator.copy(elements, data, sizeof(T) * length);
            }

            RawVec(const RawVec<T, Allocator> &other) : RawVec(other.elements, other.length)
            {
            }

            RawVec(RawVec<T, Allocator> &&other) : RawVec(other.elements, other.length)
            {
                other.elements = nullptr;
                other.length = 0;
            }

            ~RawVec()
            {
                releaseMemory();
            }

            RawVec<T, Allocator> &operator=(const RawVec<T, Allocator> &other)
            {
                Allocator allocator;
                resize(other.length);
                allocator.copy(elements, other.elements, sizeof(T) * length);
                return *this;
            }

            RawVec<T, Allocator> &operator=(RawVec<T, Allocator> &&other)
            {
                clearAll();
                elements = other.elements;
                other.elements = nullptr;
                length = other.length;
                other.length = 0;
                return *this;
            }

            T *ptr()
            {
                return elements;
            }

            const T *ptr() const
            {
                return elements;
            }

            std::size_t len() const
            {
                return length;
            }

            void resize(std::size_t newLength)
            {
                if (newLength == 0)
                {
                    deleteElements();
                }
                else if (newLength > length)
                {
                    Allocator allocator;
                    elements = static_cast<T *>(allocator.resize(elements, sizeof(T) * length, sizeof(T) * newLength));
                }
                length = newLength;
            }

            template <class Itr>
            void copy(Itr begin, std::size_t length)
            {
                resize(length);
                for (std::size_t i = 0; i < length; i++)
                {
                    copyElement(*begin, i);
                    begin++;
                }
            }

            void copyMemory(const T *ptr, std::size_t length)
            {
                Allocator allocator;
                resize(length);
                allocator.copy(elements, ptr, sizeof(T) * length);
            }

            template <class U, class CudaAllocator>
            friend class CudaRawVec;
        };

        template <class T, class CudaAllocator = Memory::DefaultCudaAllocator>
        class CudaRawVec : public RawVec<T, CudaAllocator>
        {
        private:
            virtual void copyElement(const T &element, std::size_t index) override
            {
                CudaAllocator allocator;
                allocator.copyHostToDevice(this->elements + index, &element, sizeof(T));
            }

        public:
            CudaRawVec() : RawVec<T, CudaAllocator>()
            {
            }

            CudaRawVec(std::size_t length) : RawVec<T, CudaAllocator>(length)
            {
            }

            CudaRawVec(const T *data, std::size_t length) : RawVec<T, CudaAllocator>(length)
            {
                if (length != 0)
                {
                    CudaAllocator allocator;
                    allocator.copyHostToDevice(this->elements, data, sizeof(T) * length);
                }
            }

            CudaRawVec(const CudaRawVec<T, CudaAllocator> &other) : RawVec<T, CudaAllocator>(other.elements, other.length)
            {
            }

            CudaRawVec(CudaRawVec<T, CudaAllocator> &&other) : RawVec<T, CudaAllocator>(std::move(other.elements), other.length)
            {
                other.elements = nullptr;
                other.length = 0;
            }

            CudaRawVec<T, CudaAllocator> &operator=(const CudaRawVec<T, CudaAllocator> &other)
            {
                return static_cast<CudaRawVec<T, CudaAllocator> &>(RawVec<T, CudaAllocator>::operator=(other));
            }

            CudaRawVec<T, CudaAllocator> &operator=(CudaRawVec<T, CudaAllocator> &&other)
            {
                return static_cast<CudaRawVec<T, CudaAllocator> &>(RawVec<T, CudaAllocator>::operator=(std::move(other)));
            }

            template <class Allocator = Memory::DefaultAllocator>
            RawVec<T, Allocator> toVec() const
            {
                Allocator vecAlloc;
                CudaAllocator allocator;
                T *data = vecAlloc(this->length);
                allocator.copyDeviceToHost(data, this->elements, sizeof(T) * this->length);
                return RawVec<T, Allocator>(data, this->length);
            }

            template <class Allocator>
            void copyToRawVec(RawVec<T, Allocator> &vec) const
            {
                CudaAllocator allocator;
                vec.resize(this->length);
                allocator.copyDeviceToHost(vec.elements, this->elements, sizeof(T) * this->length);
            }

            template <class Allocator>
            void copyFromRawVec(const RawVec<T, Allocator> &vec)
            {
                CudaAllocator allocator;
                this->resize(vec.length);
                allocator.copyHostToDevice(this->elements, vec.elements, sizeof(T) * vec.length);
            }
        };
    } // namespace Raw
} // namespace ChiaVec

#endif