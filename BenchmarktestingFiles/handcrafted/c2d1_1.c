#include "assert.h"

int main() {
    int x = 1;
    int y = 1;
    while (__VERIFIER_nondet_int()) {
        x = x + y;
        y = y + 1;
    }
    __VERIFIER_assert(x >= y && y >= 1);
    return 0;
}
