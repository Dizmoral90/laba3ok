#include <iostream>
#include <immintrin.h>  // Библиотека SIMD-интринсиков (SSE, AVX)
#include <chrono>       // Для замера времени
#include <cmath>        // Для fabs (модуль числа)
#include <iomanip>      // Для форматирования вывода
#include <Windows.h>    // Для настройки кодировки консоли в Windows

using namespace std;
using namespace std::chrono;

const int N = 4096; // Размерность матриц

// Инициализация матрицы случайными числами
void fillMatrix(float* mat) {
    for (int i = 0; i < N * N; ++i) {
        // Генерируем числа от 0.0 до 9.9, чтобы избежать переполнения float при умножении
        mat[i] = static_cast<float>(rand() % 100) / 10.0f;
    }
}

// 1. СКАЛЯРНОЕ УМНОЖЕНИЕ (Базовый алгоритм)
// Используем порядок циклов i-k-j для лучшего использования кэша процессора
void scalarMultiply(const float* A, const float* B, float* C) {
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < N; ++k) {
            float a_ik = A[i * N + k]; // Запоминаем элемент, чтобы не обращаться к памяти повторно
            for (int j = 0; j < N; ++j) {
                C[i * N + j] += a_ik * B[k * N + j];
            }
        }
    }
}

// 2. ВЕКТОРНОЕ УМНОЖЕНИЕ (С использованием SSE)
void sseMultiply(const float* A, const float* B, float* C) {
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < N; ++k) {
            // Берем один элемент из A и копируем его во все 4 ячейки 128-битного регистра SSE
            __m128 a_vec = _mm_set1_ps(A[i * N + k]);

            // Внутренний цикл обрабатывает по 4 элемента за один такт (шаг 4)
            for (int j = 0; j < N; j += 4) {
                // Загружаем 4 элемента из B и 4 элемента из C в регистры
                // Используем loadu/storeu (unaligned), так как память не всегда выровнена по 16 байт
                __m128 b_vec = _mm_loadu_ps(&B[k * N + j]);
                __m128 c_vec = _mm_loadu_ps(&C[i * N + j]);

                // Параллельно умножаем a_vec на b_vec и прибавляем к c_vec
                c_vec = _mm_add_ps(c_vec, _mm_mul_ps(a_vec, b_vec));

                // Сохраняем результат обратно в оперативную память
                _mm_storeu_ps(&C[i * N + j], c_vec);
            }
        }
    }
}

// Проверка корректности: сравнение двух матриц с учетом погрешности float
bool verifyResults(const float* C_scalar, const float* C_sse) {
    const float epsilon = 1e-3f; // Допустимая погрешность для одинарной точности
    for (int i = 0; i < N * N; ++i) {
        if (fabs(C_scalar[i] - C_sse[i]) > epsilon) {
            return false;
        }
    }
    return true;
}

int main() {
    // Настройка консоли Windows для корректного отображения русского текста
    SetConsoleOutputCP(CP_UTF8);
    setlocale(LC_ALL, "ru_RU.UTF-8");

    cout << "=== Лабораторная работа: SIMD-инструкции (SSE, float) ===" << endl;
    cout << "Выделение памяти под 4 матрицы 4096x4096..." << endl;
    cout << "Откройте Диспетчер задач (Ctrl+Shift+Esc) -> вкладка Подробности!" << endl;
    system("pause"); // Пауза, чтобы ты успела открыть Диспетчер задач

    // Выделяем память. Используем одномерные массивы для непрерывности данных в кэше.
    float* A = new float[N * N];
    float* B = new float[N * N];
    float* C_scalar = new float[N * N];
    float* C_sse = new float[N * N];

    // Инициализация матрицы результатов нулями
    for (int i = 0; i < N * N; ++i) {
        C_scalar[i] = 0.0f;
        C_sse[i] = 0.0f;
    }

    cout << "Заполнение матриц A и B случайными числами..." << endl;
    fillMatrix(A);
    fillMatrix(B);

    // --- СКАЛЯРНОЕ УМНОЖЕНИЕ ---
    cout << "\n[1/2] Запуск скалярного умножения..." << endl;
    auto start_scalar = high_resolution_clock::now();
    scalarMultiply(A, B, C_scalar);
    auto end_scalar = high_resolution_clock::now();
    auto duration_scalar = duration_cast<milliseconds>(end_scalar - start_scalar);
    cout << "Скалярное умножение завершено. Время: " << duration_scalar.count() << " мс." << endl;

    // --- ВЕКТОРНОЕ (SSE) УМНОЖЕНИЕ ---
    cout << "\n[2/2] Запуск векторного (SSE) умножения..." << endl;
    auto start_sse = high_resolution_clock::now();
    sseMultiply(A, B, C_sse);
    auto end_sse = high_resolution_clock::now();
    auto duration_sse = duration_cast<milliseconds>(end_sse - start_sse);
    cout << "Векторное умножение завершено. Время: " << duration_sse.count() << " мс." << endl;

    // --- АНАЛИЗ РЕЗУЛЬТАТОВ ---
    double speedup = static_cast<double>(duration_scalar.count()) / duration_sse.count();
    cout << "\n--- Сравнение быстродействия ---" << endl;
    cout << "SSE быстрее скалярного метода в " << fixed << setprecision(2) << speedup << " раз." << endl;

    cout << "\n--- Проверка корректности ---" << endl;
    if (verifyResults(C_scalar, C_sse)) {
        cout << "УСПЕХ: Результаты скалярного и векторного умножения совпадают (с учетом погрешности float)." << endl;
    }
    else {
        cout << "ОШИБКА: Результаты не совпадают." << endl;
    }

    // --- РАСЧЕТ ПАМЯТИ ---
    cout << "\n--- Расчет оперативной памяти ---" << endl;
    size_t elements = (size_t)N * N;
    size_t bytes_per_matrix = elements * sizeof(float);
    size_t total_bytes = bytes_per_matrix * 4; // 4 матрицы: A, B, C_scalar, C_sse
    cout << "Размер одной матрицы: " << bytes_per_matrix / (1024 * 1024) << " МБ" << endl;
    cout << "Общий расчетный размер (4 матрицы): " << total_bytes / (1024 * 1024) << " МБ" << endl;
    cout << "Сравните это число с колонкой 'Выделено памяти' в Диспетчере задач." << endl;

    // Освобождение памяти
    delete[] A;
    delete[] B;
    delete[] C_scalar;
    delete[] C_sse;

    system("pause");
    return 0;
}