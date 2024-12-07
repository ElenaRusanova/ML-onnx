#pragma once
#include <string>
#include <vector>

// Абстрактный базовый класс для всех моделей
class IModel {
public:
    virtual ~IModel() = default;

    // Виртуальная функция для запуска модели
    virtual void RunModel(const std::string& imageFile, const std::string& labelFile, const std::wstring& modelPath) = 0;
};


#ifdef _WIN32
#define MODEL_API __declspec(dllexport)  // Экспортируем функцию для DLL на Windows
#else
#define MODEL_API // Для других платформ экспорт не нужен
#endif