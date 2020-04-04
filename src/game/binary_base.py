import time

def make_binary(num, base=2):
    text = ""
    max_val = 1

    if num == 0:
        return "0"

    while max_val < num + 1:
        max_val *= base
    current_bit = int(max_val / base)

    while current_bit > 0:
        if num >= current_bit:
            text = text + str(num // current_bit)
            num = int(num % current_bit)
        else:
            text = text + "0"
        if current_bit == 1:
            break
        current_bit = int(current_bit / base)
    return text


# def matrix_inverse(r, m):
#     num_size = len(str(r ** m)) + 1
#     for num in range(r ** m):
#         bnum = make_binary(num, r).rjust(m, "0")
#         rev_bnum = bnum[::-1]
#         left = f"{num}".ljust(num_size, " ") + bnum
#         right = rev_bnum + f" {int(rev_bnum, r)}"
#
#         print(left + " " + right)


if __name__ == '__main__':
    for x in range(239299329230617529590083):
        out = list(str(make_binary(x, 3)))
        out.reverse()
        # print(out)
        out = ''.join(out)
        if x > 239299329230617520590083:
            print(f"{x:>7}: {out:<049}")
        if not x % 100000:
            time.sleep(0.001)

