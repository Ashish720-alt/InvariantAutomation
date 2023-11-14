#include "assert.h"

int main() {
    int x = 1;
    while (x <= 5) {
        x = x + 1;
    }
    __VERIFIER_assert(x <= 6);
    return 0;
}
