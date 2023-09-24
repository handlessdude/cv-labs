# 1. Цветовая коррекция изображений s. 18
#   1.1 Коррекция с опорным цветом
#   1.2 Серый мир
#   1.3 По виду функции преобразования
# 2. Яркостная коррекция в интерактивном режиме по виду функции преобразования (необязательное дополнительное задание)
#   2.1 График функции кусочно линейный
#   2.2 График функции интерполируется сплайном
# 3. Коррекция на основе гистограммы
#   3.1 Нормализация гистограммы
#   3.2 Эквализация гистограммы


from utils.image_hist import plot_channel_hists
from utils.fs import open_img
from resources import filename_data


dir_in = "../inputs"
dir_out = "../outputs-02"


def main():
    for data in filename_data:
        img = open_img(dir_in, data['in'])
        plot_channel_hists(img, dir_out,  data['out'])


if __name__ == '__main__':
    main()
