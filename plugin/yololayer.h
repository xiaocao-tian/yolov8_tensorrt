#pragma once
#include "NvInfer.h"
#include <string>
#include <vector>

namespace nvinfer1
{
    class __declspec(dllexport) YoloLayerPlugin : public nvinfer1::IPluginV2IOExt {
    public:
        YoloLayerPlugin(int classCount, int netWdith, int netHeight, int maxOut);
        YoloLayerPlugin(const void* data, size_t length);
        ~YoloLayerPlugin();

        int getNbOutputs() const noexcept override {
            return 1;
        }

        nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims* inputs, int nbInputDims) noexcept override;

        int initialize() noexcept override;

        virtual void terminate() noexcept override {}

        virtual size_t getWorkspaceSize(int maxBatchSize) const noexcept override { return 0; }

        virtual int enqueue(int batchSize, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

        virtual size_t getSerializationSize() const noexcept override;

        virtual void serialize(void* buffer) const noexcept override;

        bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) const noexcept override {
            return inOut[pos].format == TensorFormat::kLINEAR && inOut[pos].type == DataType::kFLOAT;
        }


        const char* getPluginType() const noexcept override;

        const char* getPluginVersion() const noexcept override;

        void destroy() noexcept override;

        IPluginV2IOExt* clone() const noexcept override;

        void setPluginNamespace(const char* pluginNamespace) noexcept override;

        const char* getPluginNamespace() const noexcept override;

        nvinfer1::DataType getOutputDataType(int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept;

        bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const noexcept override;

        bool canBroadcastInputAcrossBatch(int inputIndex) const noexcept override;

        void attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) noexcept override;

        void configurePlugin(PluginTensorDesc const* in, int32_t nbInput, PluginTensorDesc const* out, int32_t nbOutput) noexcept override;

        void detachFromContext() noexcept override;

    private:
        void forwardGpu(const float* const* inputs, float* output, cudaStream_t stream, int mYoloV8netHeight, int mYoloV8NetWidth, int batchSize);
        int mThreadCount = 256;
        const char* mPluginNamespace;
        int mClassCount;
        int mYoloV8NetWidth;
        int mYoloV8netHeight;
        int mMaxOutObject;
    };

    class __declspec(dllexport) YoloPluginCreator : public nvinfer1::IPluginCreator {
    public:
        YoloPluginCreator();
        ~YoloPluginCreator() override = default;

        const char* getPluginName() const noexcept override;

        const char* getPluginVersion() const noexcept override;

        const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;

        nvinfer1::IPluginV2IOExt* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept override;

        nvinfer1::IPluginV2IOExt* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;

        void setPluginNamespace(const char* libNamespace) noexcept override {
            mNamespace = libNamespace;
        }

        const char* getPluginNamespace() const noexcept override {
            return mNamespace.c_str();
        }

    private:
        std::string mNamespace;
        static PluginFieldCollection mFC;
        static std::vector<PluginField> mPluginAttributes;
    };
    REGISTER_TENSORRT_PLUGIN(YoloPluginCreator);
} // namespace nvinfer1

