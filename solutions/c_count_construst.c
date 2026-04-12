#include <stdio.h>
#include <string.h>
#include <stdlib.h>

char *maximum_odd_binary(const char *s)
{
    int len = strlen(s);
    int ones = 0;

    for (int i = 0; i < len; i++)
    {
        ones += (s[i] == '1');
    }

    char *result = (char *)malloc(len + 1);

    memset(result, '1', ones - 1);
    memset(result + (ones - 1), '0', len - ones);

    result[len - 1] = '1';
    result[len] = '\0';
    
    return result;
}

int main()
{
    char s[] = "0101";
    char *res = maximum_odd_binary(s);
    printf("%s\n", res);
    free(res);
    return 0;
}