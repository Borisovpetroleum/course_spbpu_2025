import matplotlib
import math as mt
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import odeint, solve_ivp
import pandas as pd
import json


def calc_ws(gamma_wat: float) -> float:
    """
    Функция для расчета солесодержания в воде

    :param gamma_wat: относительная плотность по пресной воде с плотностью 1000 кг/м3, безразм.

    :return: солесодержание в воде, г/г
    """
    ws = (
        1
        / (gamma_wat * 1000)
        * (
            1.36545 * gamma_wat * 1000
            - (3838.77 * gamma_wat * 1000 - 2.009 * (gamma_wat * 1000) ** 2) ** 0.5
        )
    )
    # если значение отрицательное, значит скорее всего плотность ниже допустимой 992 кг/м3
    if ws > 0:
        return ws
    else:
        return 0


def calc_rho_w(ws: float, t: float) -> float:
    """
    Функция для расчета плотности воды в зависимости от температуры и солесодержания

    :param ws: солесодержание воды, г/г
    :param t: температура, К

    :return: плотность воды, кг/м3
    """
    rho_w = 1000 * (1.0009 - 0.7114 * ws + 0.2605 * ws**2) ** (-1)

    return rho_w / (1 + (t - 273) * 1e-4 * (0.269 * (t - 273) ** 0.637 - 0.8))


def calc_mu_w(ws: float, t: float, p: float) -> float:
    """
    Функция для расчета динамической вязкости воды по корреляции Matthews & Russel

    :param ws: солесодержание воды, г/г
    :param t: температура, К
    :param p: давление, Па

    :return: динамическая вязкость воды, сПз
    """
    a = (
        109.574
        - (0.840564 * 1000 * ws)
        + (3.13314 * 1000 * ws**2)
        + (8.72213 * 1000 * ws**3)
    )
    b = 1.12166 - 2.63951 * ws + 6.79461 * ws**2 + 54.7119 * ws**3 - 155.586 * ws**4

    mu_w = (
        a
        * (1.8 * t - 460) ** (-b)
        * (0.9994 + 0.0058 * (p * 1e-6) + 0.6534 * 1e-4 * (p * 1e-6) ** 2)
    )
    return mu_w


def calc_n_re(rho_w: float, q_ms: float, mu_w: float, d_tub: float) -> float:
    """
    Функция для расчета числа Рейнольдса

    :param rho_w: плотность воды, кг/м3
    :param q_ms: дебит жидкости, м3/с
    :param mu_w: динамическая вязкость воды, сПз
    :param d_tub: диаметр НКТ, м

    :return: число Рейнольдса, безразмерн.
    """
    v = q_ms / (np.pi * d_tub**2 / 4)
    return rho_w * v * d_tub / mu_w * 1000


def calc_ff_churchill(n_re: float, roughness: float, d_tub: float) -> float:
    """
    Функция для расчета коэффициента трения по корреляции Churchill

    :param n_re: число Рейнольдса, безразмерн.
    :param roughness: шероховатость стен трубы, м
    :param d_tub: диаметр НКТ, м

    :return: коэффициент трения, безразмерн.
    """
    a = (-2.457 * np.log((7 / n_re) ** 0.9 + 0.27 * (roughness / d_tub))) ** 16
    b = (37530 / n_re) ** 16

    ff = 8 * ((8 / n_re) ** 12 + 1 / (a + b) ** 1.5) ** (1 / 12)
    return ff


def calc_ff_churchill(n_re: float, roughness: float, d_tub: float) -> float:
    """
    Функция для расчета коэффициента трения по корреляции Churchill

    :param n_re: число Рейнольдса, безразмерн.
    :param roughness: шероховатость стен трубы, м
    :param d_tub: диаметр НКТ, м

    :return: коэффициент трения, безразмерн.
    """
    a = (-2.457 * np.log((7 / n_re) ** 0.9 + 0.27 * (roughness / d_tub))) ** 16
    b = (37530 / n_re) ** 16

    ff = 8 * ((8 / n_re) ** 12 + 1 / (a + b) ** 1.5) ** (1 / 12)
    return ff


def calc_ff_jain(n_re: float, roughness: float, d_tub: float) -> float:
    """
    Функция для расчета коэффициента трения по корреляции Jain

    :param n_re: число Рейнольдса, безразмерн.
    :param roughness: шероховатость стен трубы, м
    :param d_tub: диаметр НКТ, м

    :return: коэффициент трения, безразмерн.
    """
    if n_re < 3000:
        ff = 64 / n_re
    else:
        ff = 1 / (1.14 - 2 * np.log10(roughness / d_tub + 21.25 / (n_re**0.9))) ** 2
    return ff


def calc_dp_dl_grav(rho_w: float, angle: float):
    """
    Функция для расчета градиента на гравитацию

    :param rho_w: плотность воды, кг/м3
    :param angle: угол наклона скважины к горизонтали, градусы

    :return: градиент давления на гравитацию в трубе, Па/м
    """
    dp_dl_grav = rho_w * 9.81 * np.sin(angle / 180 * np.pi)
    return dp_dl_grav


def calc_dp_dl_fric(
    rho_w: float, mu_w: float, q_ms: float, d_tub: float, roughness: float
):
    """
    Функция для расчета градиента давления на трение

    :param rho_w: плотность воды, кг/м3
    :param mu_w: динамическая вязкость воды, сПз
    :param q_ms: дебит жидкости, м3/с
    :param d_tub: диаметр НКТ, м
    :param roughness: шероховатость стен трубы, м

    :return: градиент давления в трубе, Па/м
    """
    if q_ms != 0:
        n_re = calc_n_re(rho_w, q_ms, mu_w, d_tub)
        ff = calc_ff_churchill(n_re, roughness, d_tub)
        dp_dl_fric = ff * rho_w * q_ms**2 / d_tub**5
    else:
        dp_dl_fric = 0
    return dp_dl_fric


def calc_dp_dl(
    rho_w: float, mu_w: float, angle: float, q_ms: float, d_tub: float, roughness: float
) -> float:
    """
    Функция для расчета градиента давления в трубе

    :param rho_w: плотность воды, кг/м3
    :param mu_w: динамическая вязкость воды, сПз
    :param angle: угол наклона скважины к горизонтали, градусы
    :param q_ms: дебит жидкости, м3/с
    :param d_tub: диаметр НКТ, м
    :param roughness: шероховатость стен трубы, м

    :return: градиент давления в трубе, Па/м
    """
    dp_dl_grav = calc_dp_dl_grav(rho_w, angle)

    dp_dl_fric = calc_dp_dl_fric(rho_w, mu_w, q_ms, d_tub, roughness)

    dp_dl = dp_dl_grav - 0.815 * dp_dl_fric

    return dp_dl


## Графики функций
# построения графика функции зависимости плотности воды от температуры
x = np.linspace(0, 100, 10)  # задание массива значений для построения графика
plt.plot(x, [calc_rho_w(0, t + 273) for t in x])
plt.title("Зависимость плотности от температуры")
plt.show()
x = np.linspace(992, 1300, 50)  # задание массива значений для построения графика
plt.plot(x, [calc_ws(gamma_wat / 1000) for gamma_wat in x])
plt.title("Зависимость солености от плотности воды")
plt.show()
x = np.linspace(0, 100, 50)
plt.plot(
    x, [calc_mu_w(0.0001, t + 273, 1 * 101325) for t in x], label="соленость 0.0001"
)
plt.plot(x, [calc_mu_w(0.001, t + 273, 1 * 101325) for t in x], label="соленость 0.001")
plt.plot(x, [calc_mu_w(0.01, t + 273, 1 * 101325) for t in x], label="соленость 0.01")
plt.plot(x, [calc_mu_w(0.1, t + 273, 1 * 101325) for t in x], label="соленость 0.1")
plt.title("Зависимость вязкости от температуры")
plt.xlabel("Температура, С")
plt.ylabel("Динамическая вязкость, СП")
plt.legend()
plt.show()
x = np.linspace(0, 5, 50)
plt.plot(x, [calc_n_re(rho_w=1000, q_ms=t / 86400, mu_w=1, d_tub=0.062) for t in x])
plt.title("Зависимость числа Рейнольдса от дебита нагнетательной скважины")
plt.xlabel("Дебит м3/сут")
plt.show()
x = np.linspace(1, 50, 300)


n_re_list = [calc_n_re(rho_w=1000, q_ms=t / 86400, mu_w=1, d_tub=0.062) for t in x]

plt.plot(
    x, [calc_ff_churchill(t, 0.0001, 0.62) for t in n_re_list], label="Расчет по Джейн"
)
plt.plot(
    x, [calc_ff_jain(t, 0.0001, 0.62) for t in n_re_list], label="Расчет по Черчилю"
)
plt.title("Зависимость коэффициента трения от дебита нагнетательной скважины")
plt.xlabel("Дебит жидкости, м3/сут")
plt.ylabel("коэффициент трения")
plt.legend()
plt.show()
# На графике зависимости коэффициента трения Муди показаны расчеты выполненные с использованием корреляций Джейна и Черчиля. Видно что корреляции хорошо совпадают друг с другом. Корреляция Черчиля описывает и ламинарный и турбулентный режимы работы
x = np.linspace(1, 500, 30)
plt.plot(
    x,
    [
        calc_dp_dl(
            rho_w=1000, mu_w=1, angle=90, q_ms=t / 86400, d_tub=0.062, roughness=0.001
        )
        for t in x
    ],
)
plt.title("Зависимость градиента давления от дебита")
plt.show()
x = np.linspace(1, 5000, 30)
plt.plot(x, [calc_dp_dl_grav(rho_w=1000, angle=90) for t in x])
plt.title("Зависимость градиента давления по гравитации от дебита")
plt.show()
x = np.linspace(1, 5000, 30)
plt.plot(
    x,
    [
        calc_dp_dl_fric(
            rho_w=1000, mu_w=1, q_ms=t / 86400, d_tub=0.062, roughness=0.001
        )
        for t in x
    ],
)
plt.title("Зависимость градиента давления по трению от дебита")
plt.show()
# Расчет распределения давления
# Для расчета необходимо задать исходные данные и перезапустить расчет
{
    "gamma_water": 1.0234161726214432,
    "md_vdp": 2658.8179653586294,
    "d_tub": 0.07038126190217257,
    "angle": 75.69418139887297,
    "roughness": 0.0003897850584166272,
    "p_wh": 107.53764429251866,
    "t_wh": 28.64158285127942,
    "temp_grad": 2.3294714876981493,
}
Q = 100  # дебит флюида, м3/cут
q_m3_sec = Q / 86400  # дебит флюида, м3/c
gamma_water = 1.0234161726214432  # относительная плотность по пресной воде с плотностью 1000 кг/м3, безразм
H = 2658.8179653586294  # измеренная глубина забоя скважины
d_tub = 0.07038126190217257  # диаметр НКТ, м
angle = 75.69418139887297  # угол наклона скважины к горизонтали, градусы
roughness = 0.0003897850584166272  # шероховатость трубы, м
p_wh = 107.53764429251866 * 101325  # давление на устье, Па
t_wh = 28.64158285127942  # температура на устье скважины, С
temp_grad = 2.3294714876981493  # геотермический градиент, К/м * (1e-2)


# Граничные условия задаются на устье скважины

from scipy.integrate import solve_ivp
import numpy as np


def __integr_func(
    h: float,
    pt: tuple,
    temp_grad: float,
    gamma_wat: float,
    angle: float,
    q_ms: float,
    d_tub: float,
    roughness: float,
) -> tuple:
    """
    Функция для интегрирования трубы

    :param h: текущая глубина, м
    :param pt: текущее давление, Па и текущая температура, К
    :param temp_grad: геотермический градиент, К/м * (1e-2)
    :param gamma_wat: относительная плотность по пресной воде с плотностью 1000 кг/м3, безразм.
    :param angle: угол наклона скважины к горизонтали, градусы
    :param q_ms: дебит жидкости, м3/с
    :param d_tub: диаметр НКТ, м
    :param roughness: шероховатость стен трубы, м

    :return: градиенты давления, Па/м и температуры, К/м
    """
    p, t = pt
    ws = calc_ws(gamma_wat)  # соленость

    rho_w = calc_rho_w(ws, t)  # Плотность при текущих условиях
    mu_w = calc_mu_w(ws, t, p)  # Вязкость при текущих условиях

    dp_dl = calc_dp_dl(rho_w, mu_w, angle, q_ms, d_tub, roughness)
    dt_dl = temp_grad * 1e-2

    return [dp_dl, dt_dl]


def calc_pipe(
    p_wh: float,
    t_wh: float,
    h0: float,
    md_vdp: float,
    temp_grad: float,
    gamma_wat: float,
    angle: float,
    q_ms: float,
    d_tub: float,
    roughness: float,
) -> tuple:
    """
    Функция для расчета давления в трубе

    :param p_wh: буферное давление, Па
    :param t_wh: температура жидкости у буферной задвижки, К
    :param h0: начальная глубина, м
    :param md_vdp: глубина верхних дыр перфорации, м
    :param temp_grad: геотермический градиент, К/м * (1e-2)
    :param gamma_wat: относительная плотность по пресной воде с плотностью 1000 кг/м3, безразм.
    :param angle: угол наклона скважины к горизонтали, градусы
    :param q_ms: дебит жидкости, м3/с
    :param d_tub: диаметр НКТ, м
    :param roughness: шероховатость стен трубы, м

    :return: давление, Па и температура, K, глубины
    """

    h_eval = np.linspace(
        h0, md_vdp, 200
    )  # равномерная сетка по глубине (от 0 до глубины забоя)

    result = solve_ivp(
        __integr_func,
        t_span=(h0, md_vdp),
        y0=[p_wh, t_wh],
        t_eval=h_eval,
        args=(temp_grad, gamma_wat, angle, q_ms, d_tub, roughness),
        method="RK45",
        dense_output=True,
    )

    p_res = result.y[0]
    t_res = result.y[1]

    return p_res, t_res, h_eval


def calc_p_wf(
    p_wh: float,
    t_wh: float,
    h0: float,
    md_vdp: float,
    temp_grad: float,
    gamma_wat: float,
    angle: float,
    q_ms: float,
    d_tub: float,
    roughness: float,
) -> float:
    """
    Функция для расчета давления на забое скважины

    :param p_wh: буферное давление, Па
    :param t_wh: температура жидкости у буферной задвижки, К
    :param h0: начальная глубина, м
    :param md_vdp: глубина верхних дыр перфорации, м
    :param temp_grad: геотермический градиент, К/м * (1e-2)
    :param gamma_wat: относительная плотность по пресной воде с плотностью 1000 кг/м3, безразм.
    :param angle: угол наклона скважины к горизонтали, градусы
    :param q_ms: дебит жидкости, м3/с
    :param d_tub: диаметр НКТ, м
    :param roughness: шероховатость стен трубы, м

    :return: давление на забое скважины, Па
    """
    p_res, t_res, h_eval = calc_pipe(
        p_wh, t_wh, h0, md_vdp, temp_grad, gamma_wat, angle, q_ms, d_tub, roughness
    )

    return p_res[-1]


results = calc_pipe(
    p_wh,
    t_wh + 273,
    h0=0,
    md_vdp=H,
    temp_grad=temp_grad,
    gamma_wat=gamma_water,
    angle=angle,
    q_ms=q_m3_sec,
    d_tub=d_tub,
    roughness=roughness,
)
results
p_res = results[0] / 101325
t_res = results[1] - 273
h_res = results[2]
p_res
t_res
h_res
# Построение графика распределения давления
plt.plot(p_res, h_res, label="давление")
plt.plot(t_res, h_res, label="температура")
plt.xlabel("P, Т")
plt.ylabel("h, длина скважины, м")
ax = plt.gca()
ax.invert_yaxis()
plt.legend()
plt.title("Распределение давления")
plt.show()
# VLP - таблица
q_range = list(range(0, 401, 10))
p_wf_range = []
for q in q_range:
    q_m3_sec = q / 86400
    p_wf = calc_p_wf(
        p_wh=p_wh,
        t_wh=t_wh + 273,
        h0=0,
        md_vdp=H,
        temp_grad=temp_grad,
        gamma_wat=gamma_water,
        angle=angle,
        q_ms=q_m3_sec,
        d_tub=d_tub,
        roughness=roughness,
    )
    p_wf_atm = p_wf / 101325
    p_wf_range.append(p_wf_atm)

plt.plot(q_range, p_wf_range, label="Pwf = f(q)", color="r", marker="o", markersize=4)
plt.xlabel("Приемистость, м3/сут")
plt.ylabel("Забойное давление, атм")
plt.title("VLP-таблица")
plt.grid(True)
plt.legend()
plt.show()
# Формирование JSON файла
import json

result = {"q_liq": q_range, "p_wf": p_wf_range}
with open("output.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=4)


# Инклинометрия скважины
def calc_sin_angle(md1: float, md2: float, incl: dict) -> float:
    """
    Расчет синуса угла с горизонталью по интерполяционной функции скважины

    Parameters
    ----------
    :param md1: measured depth 1, м
    :param md2: measured depth 2, м

    :return: синус угла к горизонтали
    """
    md = incl["md"]
    tvd = incl["tvd"]
    tube_func = interp1d(md, tvd, fill_value="extrapolate")
    return min((tube_func(md2) - tube_func(md1)) / (md2 - md1), 1)


def calc_angle(md1, incl):
    """
    Функция для расчета угла наклона трубы в точке

    :param md1: measured depth 1, м
    """

    md2 = md1 + 0.0001
    return np.degrees(np.arcsin(calc_sin_angle(md1, md2, incl)))


# траектория скважины, задается как массив измеренных глубин и значений отклонения от вертикали
h_md_ar_m = np.array([0, 50, 100, 200, 800, 1300, 1800, 2200, 2800])
h_tvd_ar_m = np.array([0, 50, 100, 200, 800, 1160, 1450, 1500, 1500])

# Подготовка данных по конструкции скважины
# удлинение от измеренной длины - для отрисовки графика
udl_m = interp1d(h_md_ar_m, h_md_ar_m - h_tvd_ar_m, kind="cubic")

# вертикальная глубина от измеренной
h_tvd_m = interp1d(h_md_ar_m, h_tvd_ar_m, kind="cubic")

# построим массив углов отклонения от вертикали
ang = np.arccos(np.diff(h_tvd_ar_m) / np.diff(h_md_ar_m))
# угол от измеренной глубины
ang_rad = interp1d(h_md_ar_m[:-1], ang)
cos_ang1_rad = lambda h: (
    (h_tvd_m(h + 1) - h_tvd_m(h))
)  # if h > 1 else  np.arccos((h_tvd_m(h+1)-h_tvd_m(h)) )


# готовим данные для отрисовки графика
h_ = np.linspace(0, H, num=500, endpoint=True)
plt.plot(udl_m(h_), h_tvd_m(h_), "-")
plt.xlabel("Отклонение от вертикали, м")
plt.ylabel("Глубина скважины , м")
plt.title("Схема траектории скважины")
ax = plt.gca()
ax.invert_yaxis()
plt.show()
plt.plot(cos_ang1_rad(h_), (h_), "-")
plt.xlabel("Косинус угла отклонения")
plt.ylabel("Глубина скважины , м")
plt.title("Косинус угла отклонения")
ax = plt.gca()
ax.invert_yaxis()
plt.show()

incl = {"md": h_md_ar_m, "tvd": h_tvd_ar_m}

angle = [calc_angle(h, incl) for h in h_]
plt.plot(angle, (h_), "-")
plt.xlabel("Угол, градусы")
plt.ylabel("Глубина скважины , м")
plt.title("Угол, градусы")
ax = plt.gca()
ax.invert_yaxis()
plt.show()


def calc_sin_angle(md1: float, md2: float, incl: dict) -> float:
    """
    Расчет синуса угла с горизонталью по интерполяционной функции скважины

    Parameters
    ----------
    :param md1: measured depth 1, м
    :param md2: measured depth 2, м

    :return: синус угла к горизонтали
    """
    md = incl["md"]
    tvd = incl["tvd"]
    tube_func = interp1d(md, tvd, fill_value="extrapolate")
    return min((tube_func(md2) - tube_func(md1)) / (md2 - md1), 1)


def calc_angle(md1, incl):
    """
    Функция для расчета угла наклона трубы в точке

    :param md1: measured depth 1, м
    """

    md2 = md1 + 0.001
    return np.degrees(np.arcsin(calc_sin_angle(md1, md2, incl)))


def __integr_func_incl(
    h: float,
    pt: tuple,
    temp_grad: float,
    gamma_wat: float,
    incl: dict,
    q_ms: float,
    d_tub: float,
    roughness: float,
) -> tuple:
    """
    Функция для интегрирования трубы

    :param h: текущая глубина, м
    :param pt: текущее давление, Па и текущая температура, К
    :param temp_grad: геотермический градиент, К/м * (1e-2)
    :param gamma_wat: относительная плотность по пресной воде с плотностью 1000 кг/м3, безразм.
    :param incl:
    :param q_ms: дебит жидкости, м3/с
    :param d_tub: диаметр НКТ, м
    :param roughness: шероховатость стен трубы, м

    :return: градиенты давления, Па/м и температуры, К/м
    """
    p, t = pt
    ws = calc_ws(gamma_wat)  # соленость

    rho_w = calc_rho_w(ws, t)  # Плотность при текущих условиях
    mu_w = calc_mu_w(ws, t, p)  # Вязкость при текущих условиях

    current_angle = calc_angle(h, incl)  # текущий угол наклона из инклинометрии

    dp_dl = calc_dp_dl(rho_w, mu_w, current_angle, q_ms, d_tub, roughness)
    dt_dl = temp_grad * 1e-2

    return [dp_dl, dt_dl]


def calc_pipe_incl(
    p_wh: float,
    t_wh: float,
    h0: float,
    md_vdp: float,
    temp_grad: float,
    gamma_wat: float,
    incl: dict,
    q_ms: float,
    d_tub: float,
    roughness: float,
) -> tuple:
    """
    Функция для расчета давления в трубе

    :param p_wh: буферное давление, Па
    :param t_wh: температура жидкости у буферной задвижки, К
    :param h0: начальная глубина, м
    :param md_vdp: глубина верхних дыр перфорации, м
    :param temp_grad: геотермический градиент, К/м * (1e-2)
    :param gamma_wat: относительная плотность по пресной воде с плотностью 1000 кг/м3, безразм.
    :param incl:
    :param q_ms: дебит жидкости, м3/с
    :param d_tub: диаметр НКТ, м
    :param roughness: шероховатость стен трубы, м

    :return: давление, Па и температура, K, глубины
    """
    h_eval = np.linspace(
        h0, md_vdp, 200
    )  # равномерная сетка по глубине (от 0 до глубины забоя)

    result = solve_ivp(
        __integr_func_incl,
        t_span=(h0, md_vdp),
        y0=[p_wh, t_wh],
        t_eval=h_eval,
        args=(temp_grad, gamma_wat, incl, q_ms, d_tub, roughness),
        method="RK45",
        dense_output=True,
    )

    p_res = result.y[0]
    t_res = result.y[1]

    return p_res, t_res, h_eval


def calc_p_wf_incl(
    p_wh: float,
    t_wh: float,
    h0: float,
    md_vdp: float,
    temp_grad: float,
    gamma_wat: float,
    incl: dict,
    q_ms: float,
    d_tub: float,
    roughness: float,
) -> float:
    """
    Функция для расчета давления на забое скважины

    :param p_wh: буферное давление, Па
    :param t_wh: температура жидкости у буферной задвижки, К
    :param h0: начальная глубина, м
    :param md_vdp: глубина верхних дыр перфорации, м
    :param temp_grad: геотермический градиент, К/м * (1e-2)
    :param gamma_wat: относительная плотность по пресной воде с плотностью 1000 кг/м3, безразм.
    :param angle: угол наклона скважины к горизонтали, градусы
    :param q_ms: дебит жидкости, м3/с
    :param d_tub: диаметр НКТ, м
    :param roughness: шероховатость стен трубы, м

    :return: давление на забое скважины, Па
    """
    p_res, t_res, h_eval = calc_pipe_incl(
        p_wh, t_wh, h0, md_vdp, temp_grad, gamma_wat, incl, q_ms, d_tub, roughness
    )

    return p_res[-1]


h_md_ar_m = np.array([0, 50, 100, 200, 800, 1300, 1800, 2200, 2500])
h_tvd_ar_m = np.array([0, 50, 100, 200, 780, 1160, 1450, 1500, 1500])
incl = {"md": h_md_ar_m, "tvd": h_tvd_ar_m}
results = calc_pipe_incl(
    p_wh,
    t_wh + 273,
    h0=0,
    md_vdp=H,
    temp_grad=temp_grad,
    gamma_wat=gamma_water,
    incl=incl,
    q_ms=q_m3_sec,
    d_tub=d_tub,
    roughness=roughness,
)
p_res = results[0] / 101325
t_res = results[1] - 273
h_res = results[2]
# Построение графика распределения давления
plt.plot(p_res, h_res, label="давление")
plt.plot(t_res, h_res, label="температура")
plt.xlabel("P, Т")
plt.ylabel("h, длина скважины, м")
ax = plt.gca()
ax.invert_yaxis()
plt.legend()
plt.title("Распределение давления")
plt.show()
# VLP - таблица с инклинометрией скважины
q_range = list(range(0, 401, 10))
p_wf_range_incl = []
for q in q_range:
    q_m3_sec = q / 86400
    p_wf_incl = calc_p_wf_incl(
        p_wh=p_wh,
        t_wh=t_wh + 273,
        h0=0,
        md_vdp=H,
        temp_grad=temp_grad,
        gamma_wat=gamma_water,
        incl=incl,
        q_ms=q_m3_sec,
        d_tub=d_tub,
        roughness=roughness,
    )
    p_wf_incl_atm = p_wf_incl / 101325
    p_wf_range_incl.append(p_wf_incl_atm)

plt.plot(
    q_range, p_wf_range_incl, label="Pwf = f(q)", color="r", marker="o", markersize=4
)
plt.xlabel("Приемистость, м3/сут")
plt.ylabel("Забойное давление, атм")
plt.title("VLP-таблица с инклинометрией скважины")
plt.grid(True)
plt.legend()
plt.show()
