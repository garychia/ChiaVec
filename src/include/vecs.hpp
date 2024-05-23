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

        void reserve(std::size_t extraElements)
        {
            if (length + extraElements > data.len())
            {
                expand(extraElements);
            }
        }

        void expand(std::size_t extraElements)
        {
            std::size_t capacity = std::max(data.len() * 2, data.len() + extraElements);
            data.resize(capacity);
        }

    public:
        Vec() : data(), length(0) {}

        Vec(std::size_t capacity) : data(capacity), length(0)
        {
        }

        Vec(const T *data, std::size_t length, bool onHost) : data(data, length, onHost), length(length)
        {
        }

        Vec(std::initializer_list<T> l) : data(l.size()), length(l.size())
        {
            data.copy(l.begin(), l.size());
        }

        template <class OtherAllocator, class OtherStorage>
        Vec(const Vec<T, OtherAllocator, OtherStorage> &other) : data(other.data), length(other.length)
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

        template <class OtherAllocator, class OtherStorage>
        Vec<T, Allocator, Storage> &operator=(const Vec<T, OtherAllocator, OtherStorage> &other)
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

        virtual T &operator[](std::size_t index)
        {
            return data.ptr()[index];
        }

        virtual const T &operator[](std::size_t index) const
        {
            return data.ptr()[index];
        }

        std::size_t len() const
        {
            return length;
        }

        std::optional<T *> get(std::size_t index)
        {
            return index < length ? std::optional<T *>(&this->data.ptr()[index]) : std::nullopt;
        }

        std::optional<const T *> getConst(std::size_t index) const
        {
            return index < length ? std::optional<const T *>(&this->data.ptr()[index]) : std::nullopt;
        }

        template <class U>
        void push(U &&element, bool onHost)
        {
            reserve(1);
            if (Allocator::AllocatesOnHost && onHost)
            {
                this->data.ptr()[length] = std::forward<U &&>(element);
            }
            else
            {
                Allocator allocator;
                allocator.copy(this->data.ptr() + length, &element, sizeof(T), Allocator::AllocatesOnHost, onHost);
            }
            length++;
        }

        virtual std::optional<T> pop()
        {
            if (length != 0)
            {
                length--;
                if (Allocator::AllocatesOnHost)
                {
                    return std::optional<T>(std::move(data.ptr()[length]));
                }
                else
                {
                    Allocator allocator;
                    T last[1];
                    allocator.copy(last, data.ptr() + length, sizeof(T), true, false);
                    return std::optional<T>(std::move(last[0]));
                }
            }
            return std::nullopt;
        }

        template <class OtherAllocator, class OtherStorage>
        void copyTo(Vec<T, OtherAllocator, OtherStorage> &vec) const
        {
            this->data.copyTo(vec.data);
            vec.length = this->length;
        }

        template <class OtherAllocator, class OtherStorage>
        void copyFrom(const Vec<T, OtherAllocator, OtherStorage> &vec)
        {
            this->data.copyFrom(vec.data);
            this->length = vec.length;
        }

        template <class OtherAllocator = Memory::DefaultAllocator, class OtherStorage = Raw::RawVec<T, OtherAllocator>>
        Vec<T, OtherAllocator, OtherStorage> clone() const
        {
            Vec<T, OtherAllocator, OtherStorage> vec(this->length);
            this->data.copyTo(vec.data);
            vec.length = this->length;
            return vec;
        }

        template <class U, class OtherAllocator, class OtherStorage>
        friend class Vec;

        template <class U, class OtherAllocator, class OtherStorage>
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

        template <class Fn>
        CudaVec<T, CudaAllocator, Storage> calculate(const CudaVec<T, CudaAllocator, Storage> &other, Fn deviceFunc) const
        {
            std::size_t length = std::min(this->len(), other.len());
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
    };
} // namespace ChiaVec
#endif