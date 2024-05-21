#ifndef __CHIAVEC_VECS_HPP__
#define __CHIAVEC_VECS_HPP__

#include "rawvecs.hpp"

#include <optional>

namespace ChiaVec
{
    template <class T, class Allocator = Memory::DefaultAllocator, class Storage = Raw::RawVec<T, Allocator>>
    class Vec
    {
    private:
        Storage data;
        std::size_t length;

        void reserve(std::size_t elements) {
            if (length + elements > data.len()) {
                expand(elements);
            }
        }

        void expand(std::size_t elements) {
            std::size_t capacity = std::max(data.len() * 2, elements);
            data.resize(capacity);
        }

    public:
        Vec() : data(), length(0) {}

        Vec(std::size_t capacity) : data(capacity), length(0)
        {
        }

        Vec(const T *data, std::size_t length) : data(data, length), length(length)
        {
        }

        Vec(std::initializer_list<T> l) : data(l.size()), length(l.size())
        {
            data.copy(l.begin(), l.size());
        }

        Vec(const Vec<T, Allocator, Storage> &other) : data(other.data), length(other.length)
        {
        }

        Vec(Vec<T, Allocator, Storage> &&other) : data(std::move(other.data)), length(other.length)
        {
            other.length = 0;
        }

        Vec<T, Allocator, Storage> &operator=(const Vec<T, Allocator, Storage> &other)
        {
            this->data = other.data;
            this->length = other.length;
            return *this;
        }

        Vec<T, Allocator, Storage> &operator=(Vec<T, Allocator, Storage> &&other)
        {
            this->data = std::move(other.data);
            this->length = other.length;
            return *this;
        }

        virtual std::optional<T *> operator[](std::size_t index)
        {
            return index < length ? std::optional<T *>(&data.ptr()[index]) : std::nullopt;
        }

        virtual std::optional<const T *> operator[](std::size_t index) const
        {
            return index < length ? std::optional<const T *>(&data.ptr()[index]) : std::nullopt;
        }

        std::size_t len() const
        {
            return length;
        }

        template <class U>
        void push(U &&element)
        {
            reserve(1);
            this->data.ptr()[length] = std::forward<U &&>(element);
            length++;
        }

        virtual std::optional<T> pop()
        {
            if (length != 0)
            {
                length--;
                return std::optional<T>(std::move(data.ptr()[length]));
            }
            return std::nullopt;
        }

        template <class U, class CudaAllocator, class CudaStorage>
        friend class CudaVec;
    };

    template <class T, class CudaAllocator = Memory::DefaultCudaAllocator, class Storage = Raw::CudaRawVec<T, CudaAllocator>>
    class CudaVec : public Vec<T, CudaAllocator, Storage>
    {
    public:
        using Vec<T, CudaAllocator, Storage>::Vec;

        CudaVec(const CudaVec &other) : Vec<T, CudaAllocator, Storage>(other)
        {
        }

        CudaVec(CudaVec &&other) : Vec<T, CudaAllocator, Storage>(std::move(other))
        {
        }

        CudaVec<T, CudaAllocator, Storage> &operator=(const CudaVec<T, CudaAllocator, Storage> &other)
        {
            return static_cast<CudaVec<T, CudaAllocator, Storage> &>(Vec<T, CudaAllocator, Storage>::operator=(other));
        }

        CudaVec<T, CudaAllocator, Storage> &operator=(CudaVec<T, CudaAllocator, Storage> &&other)
        {
            return static_cast<CudaVec<T, CudaAllocator, Storage> &>(Vec<T, CudaAllocator, Storage>::operator=(std::move(other)));
        }

        virtual std::optional<T *> operator[](std::size_t index) override
        {
            return std::nullopt;
        }

        virtual std::optional<const T *> operator[](std::size_t index) const override
        {
            return std::nullopt;
        }

        void push(const T &element)
        {
            CudaAllocator allocator;
            this->reserve(1);
            allocator.copyHostToDevice(this->data.ptr() + this->length, &element, sizeof(T));
            this->length++;
        }

        virtual std::optional<T> pop() override
        {
            if (this->length != 0)
            {
                CudaAllocator allocator;
                T last[1];
                this->length--;
                allocator.copyDeviceToHost(last, this->data.ptr() + this->length, 1);
                return last[0];
            }
            return std::nullopt;
        }

        template <class Fn>
        CudaVec<T, CudaAllocator, Storage> calculate(const CudaVec<T, CudaAllocator, Storage> &other, Fn deviceFunc) const
        {
            std::size_t length = std::min(this->length, other.length);
            CudaVec<T, CudaAllocator, Storage> result(length);
            deviceFunc(result.data.ptr(), this->data.ptr(), other.data.ptr(), length);
            result.length = length;
            return result;
        }

        template <class Fn>
        void calculateInplace(const CudaVec<T, CudaAllocator, Storage> &other, Fn deviceFunc)
        {
            deviceFunc(this->data.ptr(), this->data.ptr(), other.data.ptr(), std::min(this->length, other.length));
        }

        template <class Allocator, class VecStorage>
        void copyToVec(Vec<T, Allocator, VecStorage> &vec) const
        {
            this->data.copyToRawVec(vec.data);
            vec.length = this->length;
        }

        template <class Allocator, class VecStorage>
        void copyFromVec(const Vec<T, Allocator, VecStorage> &vec)
        {
            this->data.copyFromRawVec(vec.data);
            this->length = vec.length;
        }

        template <class Allocator = Memory::DefaultAllocator, class VecStorage = Raw::RawVec<T, Allocator>>
        Vec<T, Allocator, VecStorage> toVec() const
        {
            Vec<T, Allocator, VecStorage> vec(this->length);
            this->data.copyToRawVec(vec.data);
            vec.length = this->length;
            return vec;
        }

        template <class Allocator, class VecStorage>
        static CudaVec<T, CudaAllocator, Storage> fromVec(const Vec<T, Allocator, VecStorage> &vec)
        {
            CudaVec<T, CudaAllocator, Storage> cudaVec(vec.length);
            cudaVec.data.copyFromRawVec(vec.data);
            cudaVec.length = vec.length;
            return cudaVec;
        }
    };
} // namespace ChiaVec
#endif