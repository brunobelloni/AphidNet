import sys


def ProgressBar(name, value, endvalue, bar_length=50, width=10):
    percent = float(value) / endvalue
    arrow = '-' * int(round(percent * bar_length) - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))

    sys.stdout.write("\r{0: <{1}} : [{2}] {3}%".format(name, width,
                                                       arrow + spaces,
                                                       int(round(percent * 100))))

    sys.stdout.flush()

    if value == endvalue:
        sys.stdout.write('\n\n ')
