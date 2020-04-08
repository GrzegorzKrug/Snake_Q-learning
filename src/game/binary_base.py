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


if __name__ == '__main__':
    for x in range(239299329230617529590083):  # this is big number :D
        out = list(str(make_binary(x, 3)))
        out.reverse()
        # print(out)
        out = ''.join(out)
        if x > 239299329230617520590083:
            print(f"{x:>7}: {out:<049}")
        if not x % 100000:
            time.sleep(0.001)

