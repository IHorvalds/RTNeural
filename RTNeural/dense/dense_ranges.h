//
// Created by Tudor Croitoru on 30/07/2023.
//

#ifndef DENSERANGES_H_INCLUDED
#define DENSERANGES_H_INCLUDED

#include "../Layer.h"
#include <range/v3/all.hpp>
#include <span>
#include <concepts>
#include <string>

namespace RTNeural
{

template <typename Numeric, typename Tuple>
struct AddMultiply
{
    constexpr Numeric operator()(Tuple&& t, Numeric&& e)
    {
        Numeric res = {};
        std::apply([&res](const auto& ...el)
            {
                res = (el * ...) * (Numeric)1;
            }, t);
        return e + res;
    }
};

/**
 * Dynamic implementation of a fully-connected (dense) layer,
 * with no activation.
 */
template <typename T>
class Dense : public Layer<T>
{
public:
    Dense(int in_size, int out_size)
        : Layer<T>(in_size, out_size)
    {
        weights.resize(out_size);
        bias.resize(out_size);
        inVec.resize(in_size);
        outVec.resize(out_size);

        for (auto& w : weights)
        {
            w.resize(in_size);
            ranges::fill(w, (T)0);
        }

        ranges::fill(bias, (T)0);
    }

    Dense(std::initializer_list<int> sizes)
        : Dense(*sizes.begin(), *(sizes.begin() + 1))
    {}

    Dense& operator=(const Dense& other)
    {
        return *this = Dense(other);
    }

    virtual ~Dense() = default;

    /** Returns the name of this layer */
    std::string getName() const noexcept override { return "dense"; }

    /** Performs forward propagation for this layer */
    inline void forward(const T* input, T* output) noexcept override
    {
        std::span<const T> inputSpan(input, Layer<T>::in_size);
        std::span<T> outputSpan(output, Layer<T>::out_size);

        ranges::transform(ranges::views::zip(weights, ranges::views::repeat(inputSpan), bias), outputSpan.begin(),
            [](auto els)
            {
                return ranges::fold_right(
                    ranges::views::zip(std::get<0>(els), std::get<1>(els)),
                    std::get<2>(els),
                    AddMultiply<T, std::tuple<T, T>>{});
            });
    }

    /**
     * Sets the layer weights from a given vector.
     *
     * The dimension of the weights vector must be
     * weights[out_size][in_size]
     */
    void setWeights(const std::vector<std::vector<T>>& newWeights)
    {
        weights = newWeights;
    }

    /**
     * Sets the layer weights from a given array.
     *
     * The dimension of the weights array must be
     * weights[out_size][in_size]
     */
    void setWeights(T** newWeights)
    {
        for(int i = 0; i < Layer<T>::out_size; ++i)
            memcpy(weights[i].data(), newWeights[i], Layer<T>::in_size);
    }

    /**
     * Sets the layer bias from a given array of size
     * bias[out_size]
     */
    void setBias(const T* b)
    {
        memcpy(bias.data(), b, Layer<T>::out_size);
    }

    /** Returns the weights value at the given indices. */
    T getWeight(int i, int k) const noexcept { return weights[i][k]; }

    /** Returns the bias value at the given index. */
    T getBias(int i) const noexcept { return bias[i]; }

private:
    std::vector<std::vector<T>> weights;
    std::vector<T> bias;
    std::vector<T> inVec;
    std::vector<T> outVec;
};

//====================================================
/**
 * Static implementation of a fully-connected (dense) layer,
 * with no activation.
 */
template <typename T, int in_sizet, int out_sizet>
class DenseT
{
    using out_vec_type = T[out_sizet];
    using in_vec_type = T[in_sizet];
    using mat_type = T[out_sizet][in_sizet];

public:
    static constexpr auto in_size = in_sizet;
    static constexpr auto out_size = out_sizet;

    DenseT()
    {
        for (size_t i = 0; i < out_size; ++i)
        {
            memset(weights[i], 0, in_size);
        }
        memset(bias, 0, out_size);
    }

    /** Returns the name of this layer. */
    std::string getName() const noexcept { return "dense"; }

    /** Returns false since dense is not an activation layer. */
    constexpr bool isActivation() const noexcept { return false; }

    /** Reset is a no-op, since Dense does not have state. */
    void reset() { }

    /** Performs forward propagation for this layer. */
    inline void forward(const in_vec_type & ins) noexcept
    {
        std::span inputSpan(ins, in_size);
        std::span outputSpan(outs, out_size);

        ranges::transform(ranges::views::zip(weights, ranges::views::repeat(inputSpan), bias), outputSpan.begin(),
            [](auto els)
            {
                return ranges::fold_right(
                    ranges::views::zip(std::get<0>(els), std::get<1>(els)),
                    std::get<2>(els),
                    AddMultiply<T, std::tuple<T, T>>{});
            });
    }

    /**
     * Sets the layer weights from a given vector.
     *
     * The dimension of the weights vector must be
     * weights[out_size][in_size]
     */
    void setWeights(const std::vector<std::vector<T>>& newWeights)
    {
        for(int i = 0; i < out_size; ++i)
            memcpy(weights[i], newWeights[i].data(), in_size);
    }

    /**
     * Sets the layer weights from a given vector.
     *
     * The dimension of the weights array must be
     * weights[out_size][in_size]
     */
    void setWeights(T** newWeights)
    {
        for(int i = 0; i < out_size; ++i)
            memcpy(weights[i], newWeights[i], in_size);
    }

    /**
     * Sets the layer bias from a given array of size
     * bias[out_size]
     */
    void setBias(const T* b)
    {
        memcpy(bias, b, out_size);
    }

    T outs alignas(RTNEURAL_DEFAULT_ALIGNMENT)[out_size];

private:
    mat_type weights;
    out_vec_type bias;
};

}

#endif
