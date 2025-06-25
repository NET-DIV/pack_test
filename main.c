#include <stdio.h>
#include <time.h>
#include <stdlib.h>

float train[][2] = {
    {0, 0},
    {1, 2},
    {2, 4},
    {3, 6},
    {4, 8},
};
#define train_cout (sizeof(train) / sizeof(train[0]))

float rand_float(void) {
    return (float) rand() / (float) RAND_MAX;
}

float cost(float w) {
    float result = 0.0f;
    for (size_t i = 0; i < train_cout; i++) {
        float x = train[i][0];
        float y = x * w;
        float d = y - train[i][1];
        result += d * d;
    }
    result /= train_cout;
    return result;
}

int main() {
    srand(69);
    float w = 1.0f;
    float eps = 1e-3;
    float rate = 1e-3;

    for (size_t i = 0; i <500; i++) {
        float dcost = (cost(w + eps) - cost(w)) / eps;
        w -= rate * dcost;
        printf("%f, %f\n", cost(w), w);
    }

    printf("---------------------------\n");
    printf("%f", w);
}