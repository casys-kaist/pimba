from .figure3 import draw_figure3
from .figure4 import draw_figure4
from .figure5 import draw_figure5
from .figure6 import draw_figure6
from .figure12 import draw_figure12
from .figure13 import draw_figure13
from .figure14 import draw_figure14
from .figure16 import draw_figure16
from .table2 import draw_table2


def draw(exp_name_list: list[str]):
    for name in exp_name_list:
        match name:
            case "figure3":
                draw_figure3()
            case "figure4":
                draw_figure4()
            case "figure5":
                draw_figure5()
            case "figure6":
                draw_figure6()
            case "figure12":
                draw_figure12()
            case "figure13":
                draw_figure13()
            case "figure14":
                draw_figure14()
            case "figure16":
                draw_figure16()
            case "table2":
                draw_table2()
