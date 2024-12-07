#include "Resnet.h"
#include <iostream>
#include <algorithm>

void ResnetModel::RunModel(const std::string& imageFile, const std::string& labelFile, const std::wstring& modelPath) {
    Ort::Env env;
    Ort::RunOptions runOptions;
    Ort::Session session(nullptr);

    constexpr int64_t numChannels = 3;  // Цветное изображение
    constexpr int64_t width = 224;
    constexpr int64_t height = 224;
    constexpr int64_t numClasses = 1000;  // 1000 классов

    // Загрузка меток классов
    std::unordered_map<int, std::string> labels;
    int index = 0;
    for (const auto& label : Helpers::loadLabels(labelFile)) {
        labels[index++] = label;
    }

    if (labels.empty()) {
        std::cerr << "Failed to load labels: " << labelFile << std::endl;
        return;
    }

    // Загрузка изображения
    const std::vector<float> imageVec = Helpers::loadImageRGB(imageFile, width, height);
    if (imageVec.empty()) {
        std::cerr << "Failed to load or process image." << std::endl;
        return;
    }

    //------------------------------------------------------------------------------------
    
    //// Use CUDA GPU
    //Ort::SessionOptions ort_session_options;

    //OrtCUDAProviderOptions options;
    //options.device_id = 0;
    // OrtSessionOptionsAppendExecutionProvider_CUDA(ort_session_options, options.device_id);
    
    //// create session for GPU
    // session = Ort::Session(env, modelPath, ort_session_options);
    
    //------------------------------------------------------------------------------------


    // Создание сессии ONNX Runtime
    session = Ort::Session(env, modelPath.c_str(), Ort::SessionOptions{ nullptr });

    // Запуск модели
    std::array<float, numClasses> results;
    const std::array<int64_t, 4> inputShape = { 1, numChannels, height, width };
    const std::array<int64_t, 2> outputShape = { 1, numClasses };

    std::array<float, numChannels* height* width> input;
    std::copy(imageVec.begin(), imageVec.end(), input.begin());

    // Тензоры для входа и выхода
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    auto inputTensor = Ort::Value::CreateTensor<float>(memory_info, input.data(), input.size(), inputShape.data(), inputShape.size());
    auto outputTensor = Ort::Value::CreateTensor<float>(memory_info, results.data(), results.size(), outputShape.data(), outputShape.size());

    // Имя входного и выходного тензоров
    Ort::AllocatorWithDefaultOptions allocator;
    auto inputName = session.GetInputNameAllocated(0, allocator);
    auto outputName = session.GetOutputNameAllocated(0, allocator);

    const std::array<const char*, 1> inputNames = { inputName.get() };
    const std::array<const char*, 1> outputNames = { outputName.get() };

    // Выполнение модели
    session.Run(runOptions, inputNames.data(), &inputTensor, 1, outputNames.data(), &outputTensor, 1);

    // Обработка результатов
    std::vector<std::pair<int, float>> indexValuePairs;
    for (size_t i = 0; i < results.size(); ++i) {
        indexValuePairs.emplace_back(i, results[i]);
    }

    std::sort(indexValuePairs.begin(), indexValuePairs.end(), [](const auto& lhs, const auto& rhs) { return lhs.second > rhs.second; });

    std::unordered_map<std::string, float> classProbabilities;
    for (const auto& result : indexValuePairs) {
        classProbabilities[labels[result.first]] = result.second;
    }

    for (size_t i = 0; i < std::min<size_t>(5, indexValuePairs.size()); ++i) {
        const auto& result = indexValuePairs[i];
        std::cout << "Class: " << labels[result.first] << ", Probability: " << result.second << std::endl;
    }
}