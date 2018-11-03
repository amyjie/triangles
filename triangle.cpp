#include "triangle.h"

/* Mark important _byte_ offsets */
size_t BG_COLOR_OFFSET = 0;
size_t BG_COLOR_SIZE = 4;
size_t TRIANGLE_LIST_BEGIN = BG_COLOR_SIZE;
size_t TRIANGLE_SIZE = sizeof(Triangle);

/* Check if two triangles are the same. Ignores "visible" trait. */
bool operator==(const Triangle& lhs, const Triangle& rhs)
{
    return  lhs.x1 == rhs.x1 &&
            lhs.y1 == rhs.y1 &&

            lhs.x2 == rhs.x2 &&
            lhs.y2 == rhs.y2 &&

            lhs.x3 == rhs.x3 &&
            lhs.y3 == rhs.y3 &&

            lhs.r == rhs.r &&
            lhs.b == rhs.b &&
            lhs.g == rhs.g &&
            lhs.a == rhs.a;
}
