#include "Emotion.h"
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <unordered_map>
#include <algorithm>

void EmotionModel::RunModel(const std::string& imageFile, const std::string& labelFile, const std::wstring& modelPath) {
    Ort::Env env;
    Ort::RunOptions runOptions;
    Ort::Session session(nullptr);


    constexpr int64_t numChannels = 1;  // Изменено на 1 канал для черно-белого изображения
    constexpr int64_t width = 64;       // Изменено на 64
    constexpr int64_t height = 64;      // Изменено на 64
    constexpr int64_t numClasses = 8;   // Количество классов эмоций
    constexpr int64_t numInputElements = numChannels * height * width;

    // Загрузка меток эмоций
    std::unordered_map<int, std::string> labels;
    int index = 0;
    for (const auto& label : Helpers::loadLabels(labelFile)) {
        labels[index++] = label;
    }
    if (labels.empty()) {
        std::cerr << "Failed to load labels: " << labelFile << std::endl;
        return;
    }

    // Загрузка и предобработка изображения
    const std::vector<float> imageVec = Helpers::loadImage(imageFile, width, height);  // Убедитесь, что это 64x64, 1 канал
    if (imageVec.empty() || imageVec.size() != numInputElements) {
        std::cerr << "Invalid image format or failed to load image." << std::endl;
        return;
    }

    // Создание сессии ONNX Runtime для модели FER+
    session = Ort::Session(env, modelPath.c_str(), Ort::SessionOptions{ nullptr });

    // Определение формы входных и выходных тензоров
    const std::array<int64_t, 4> inputShape = { 1, numChannels, height, width };
    const std::array<int64_t, 2> outputShape = { 1, numClasses };

    // Определение массивов для входных данных и результатов
    std::array<float, numInputElements> input;
    std::array<float, numClasses> results;

    // Заполнение входного массива
    std::copy(imageVec.begin(), imageVec.end(), input.begin());

    // Создание тензоров
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    auto inputTensor = Ort::Value::CreateTensor<float>(memory_info, input.data(), input.size(), inputShape.data(), inputShape.size());
    auto outputTensor = Ort::Value::CreateTensor<float>(memory_info, results.data(), results.size(), outputShape.data(), outputShape.size());

    // Получение имён входных и выходных слоёв
    Ort::AllocatorWithDefaultOptions allocator;
    auto inputName = session.GetInputNameAllocated(0, allocator);
    auto outputName = session.GetOutputNameAllocated(0, allocator);

    const std::array<const char*, 1> inputNames = { inputName.get() };
    const std::array<const char*, 1> outputNames = { outputName.get() };

    // Запуск модели
    session.Run(runOptions, inputNames.data(), &inputTensor, 1, outputNames.data(), &outputTensor, 1);

    // Обработка результатов
    std::vector<std::pair<int, float>> indexValuePairs;
    for (size_t i = 0; i < results.size(); ++i) {
        indexValuePairs.emplace_back(i, results[i]);
    }

    // Сортировка и вывод топ-5 предсказаний
    std::sort(indexValuePairs.begin(), indexValuePairs.end(), [](const auto& lhs, const auto& rhs) { return lhs.second > rhs.second; });


    // Создаем хэш-таблицу с названиями эмоций и их вероятностями
    std::unordered_map<std::string, float> emotionProbabilities;
    for (const auto& result : indexValuePairs) {
        // Используем `labels[result.first]` для получения названия эмоции по индексу
        emotionProbabilities[labels[result.first]] = result.second;
    }

    for (size_t i = 0; i < std::min<size_t>(5, indexValuePairs.size()); ++i) {
        const auto& result = indexValuePairs[i];
        std::cout << "Emotion: " << labels[result.first] << ", Probability: " << result.second << std::endl;
    }
}