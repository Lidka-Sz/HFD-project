# funkcja plot_positions_ma

# funkcja rysująca framework strategii 1MA
# dla pojedynczego dnia

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.dates as mdates

def plot_positions_ma(data_plot,      # DataFrame z indeksem DatetimeIndex
                      date_plot,      # Data jako string 'YYYY-MM-DD'
                      col_price,      # nazwa kolumny z ceną
                      col_ma,         # kolumna z MA/medianą
                      col_pos,        # kolumna z pozycją (-1, 0, 1)
                      title,          # tytuł wykresu
                      save_graph = False,
                      width = 10,
                      height = 6,
                      file_name = None):
    """
    Rysuje wykres cen z MA i pozycjami (tło kolorowe).
    """

    # Filtrowanie danych do podanej daty
    data_day = data_plot.loc[data_plot.index.date == pd.to_datetime(date_plot).date()].copy()
    data_day = data_day[[col_price, col_ma, col_pos]]

    # Dodajemy kolumnę czasu i next_time (do prostokątów)
    data_day = data_day.reset_index()
    data_day.rename(columns={data_day.columns[0]: 'Time'}, inplace=True)
    data_day["next_Time"] = data_day["Time"].shift(-1)
    data_day["position"] = data_day[col_pos]

    # Kolory pozycji
    pos_colors = {
        -1: 'red',
         0: 'gray',
         1: 'green'
    }

    # 1. Inicjalizacja wykresu
    fig, ax = plt.subplots(figsize=(width, height))

    # 2. Najpierw rysujemy linie (cena i MA)
    ax.plot(data_day["Time"], data_day[col_price], color='black', label='Cena')
    ax.plot(data_day["Time"], data_day[col_ma], color='blue', label='MA', linewidth=1.5)

    # 3. Dopiero teraz pobieramy granice osi Y
    ymin, ymax = ax.get_ylim()

    # 4. Dodajemy kolorowe tło dla pozycji
    for i, row in data_day.iterrows():
        if pd.isna(row['position']) or pd.isna(row['next_Time']):
            continue
        color = pos_colors.get(int(row['position']), 'white')
        ax.add_patch(
            Rectangle((row['Time'], ymin),
                    row['next_Time'] - row['Time'],
                    ymax - ymin,
                    facecolor=color, alpha=0.2)
        )

    # Rysujemy wykresy linii
    ax.plot(data_day["Time"], data_day[col_price], color='black', label='Cena')
    ax.plot(data_day["Time"], data_day[col_ma], color='blue', label='MA', linewidth=1.5)

    # Formatowanie osi
    ax.set_title(title)
    ax.set_xlabel("Czas")
    ax.set_ylabel("Cena / Sygnał pozycji")
   # Usuń duplikaty w legendzie (jeśli się powtórzyły)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))  # unikalne: etykieta → linia
    ax.legend(by_label.values(), by_label.keys(), loc='upper left') 
    # ax.grid(True)
    plt.xticks(rotation=0)

    # Formatowanie osi X – tylko czas (HH:MM)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=30))

    # Zapis wykresu jeśli trzeba
    if save_graph:
        if file_name is None:
            raise ValueError("Musisz podać file_name jeśli save_graph=True")
        plt.savefig(file_name, bbox_inches='tight', dpi=300)

    plt.tight_layout()
    plt.show()
    return ax


#-----------------------------------------------
# funkcja plot_positions_2mas

# funkcja rysująca framework strategii 2MAs
# dla pojedynczego dnia

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.dates as mdates

def plot_positions_2mas(data_plot,      # DataFrame z indeksem DatetimeIndex
                        date_plot,      # Data jako string 'YYYY-MM-DD'
                        col_price,      # nazwa kolumny z ceną
                        col_fma,        # kolumna z fast moving average
                        col_sma,        # kolumna z slow moving average
                        col_pos,        # kolumna z pozycją (-1, 0, 1)
                        title,          # tytuł wykresu
                        save_graph  =False,
                        width = 10,
                        height = 6,
                        file_name = None):
    """
    Rysuje wykres ceny, dwóch średnich (fma i sma) oraz pozycje jako kolorowe tło.
    """

    # Filtrowanie danych do podanego dnia
    data_day = data_plot.loc[data_plot.index.date == pd.to_datetime(date_plot).date()].copy()
    data_day = data_day[[col_price, col_fma, col_sma, col_pos]].copy()

    # Przygotowanie kolumn czasowych
    data_day = data_day.reset_index()
    data_day.rename(columns={data_day.columns[0]: 'Time'}, inplace=True)
    data_day["next_Time"] = data_day["Time"].shift(-1)
    data_day["position"] = data_day[col_pos]

    # Kolory tła w zależności od pozycji
    pos_colors = {
        -1: 'red',
         0: 'gray',
         1: 'green'
    }

    # Inicjalizacja wykresu
    fig, ax = plt.subplots(figsize=(width, height))

    # Rysujemy linie: cena, FMA, SMA
    ax.plot(data_day["Time"], data_day[col_price], color='gray', label='Cena')
    ax.plot(data_day["Time"], data_day[col_fma], color='blue', label='Fast MA', linewidth=1.5)
    ax.plot(data_day["Time"], data_day[col_sma], color='darkgreen', label='Slow MA', linewidth=1.5)

    # Pobierz aktualne granice osi Y po narysowaniu linii
    ymin, ymax = ax.get_ylim()

    # Dodaj prostokąty jako tło dla pozycji
    for _, row in data_day.iterrows():
        if pd.isna(row['position']) or pd.isna(row['next_Time']):
            continue
        color = pos_colors.get(int(row['position']), 'white')
        ax.add_patch(
            Rectangle((row['Time'], ymin),
                      row['next_Time'] - row['Time'],
                      ymax - ymin,
                      facecolor=color,
                      alpha=0.2)
        )

    # Formatowanie osi i legenda
    ax.set_title(title)
    ax.set_xlabel("Czas")
    ax.set_ylabel("Cena / Pozycja")

    # Usuń duplikaty w legendzie
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper left')

    # Oś X – format czasu
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    # Ticki co 30 minut, zaczynające się od pierwszego czasu
    start = data_day["Time"].iloc[0]
    end = data_day["Time"].iloc[-1]
    tick_locs = pd.date_range(start=start, end=end, freq='30min')
    ax.set_xticks(tick_locs)
    ax.set_xlim(start, end)

    # Zapis do pliku, jeśli wymagane
    if save_graph:
        if file_name is None:
            raise ValueError("Musisz podać file_name jeśli save_graph=True")
        plt.savefig(file_name, bbox_inches='tight', dpi=300)

    plt.tight_layout()
    plt.show()
    return ax



#-----------------------------------------------------
# funkcja plot_positions_vb

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.dates as mdates

def plot_positions_vb(data_plot,       # DataFrame z indeksem DatetimeIndex
                      date_plot,       # Data jako string 'YYYY-MM-DD'
                      col_signal,      # kolumna z sygnałem głównym
                      col_upper,       # kolumna z górnym ograniczeniem
                      col_lower,       # kolumna z dolnym ograniczeniem
                      col_pos,         # kolumna z pozycją (-1, 0, 1)
                      title,           # tytuł wykresu
                      save_graph = False,
                      width = 10,
                      height = 6,
                      file_name = None):
    """
    Rysuje wykres sygnału z przedziałem (upper/lower) i tłem wg pozycji.
    """

    # Filtrowanie danych do podanego dnia
    data_day = data_plot.loc[data_plot.index.date == pd.to_datetime(date_plot).date()].copy()
    data_day = data_day[[col_signal, col_upper, col_lower, col_pos]].copy()

    # Przygotowanie kolumn czasowych
    data_day = data_day.reset_index()
    data_day.rename(columns={data_day.columns[0]: 'Time'}, inplace=True)
    data_day["next_Time"] = data_day["Time"].shift(-1)
    data_day["position"] = data_day[col_pos]

    # Kolory pozycji
    pos_colors = {
        -1: 'red',
         0: 'gray',
         1: 'green'
    }

    # Inicjalizacja wykresu
    fig, ax = plt.subplots(figsize=(width, height))

    # Rysujemy sygnał i przedział (dolna i górna linia)
    ax.plot(data_day["Time"], data_day[col_signal], color='black', label='Sygnał')
    ax.plot(data_day["Time"], data_day[col_upper], color='blue', linewidth=1.5, label='Upper Bound')
    ax.plot(data_day["Time"], data_day[col_lower], color='blue', linewidth=1.5, label='Lower Bound')

    # Ustawiamy zakres Y po narysowaniu linii
    ymin, ymax = ax.get_ylim()

    # Tło według pozycji
    for _, row in data_day.iterrows():
        if pd.isna(row['position']) or pd.isna(row['next_Time']):
            continue
        color = pos_colors.get(int(row['position']), 'white')
        ax.add_patch(
            Rectangle((row['Time'], ymin),
                      row['next_Time'] - row['Time'],
                      ymax - ymin,
                      facecolor=color,
                      alpha=0.2)
        )

    # Formatowanie osi i legenda
    ax.set_title(title)
    ax.set_xlabel("Czas")
    ax.set_ylabel("Sygnał / Pozycja")

    # Usuwanie duplikatów w legendzie
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper left')

    # Formatowanie osi X: tylko godzina:minuta
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    # Ticki co 15 minut od początku danych
    start = data_day["Time"].iloc[0]
    end = data_day["Time"].iloc[-1]
    tick_locs = pd.date_range(start=start, end=end, freq='15min')
    ax.set_xticks(tick_locs)
    ax.set_xlim(start, end)
    plt.xticks(rotation=45)

    # Zapis wykresu do pliku, jeśli wymagany
    if save_graph:
        if file_name is None:
            raise ValueError("Musisz podać file_name jeśli save_graph=True")
        plt.savefig(file_name, bbox_inches='tight', dpi=300)

    plt.tight_layout()
    plt.show()
    return ax
    
#-------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.dates as mdates

def plot_positions_2vb(data_plot,          # DataFrame z DatetimeIndex
                       date_plot,          # Data jako string 'YYYY-MM-DD'
                       col_signal,         # kolumna z sygnałem
                       col_upper_entry,    # górna bariera wejścia
                       col_upper_exit,     # górna bariera wyjścia
                       col_lower_entry,    # dolna bariera wejścia
                       col_lower_exit,     # dolna bariera wyjścia
                       col_pos,            # kolumna z pozycją (-1, 0, 1)
                       title,              # tytuł wykresu
                       save_graph=False,
                       width=10,
                       height=6,
                       file_name=None):
    """
    Rysuje sygnał i przedziały strategii 2VB oraz tło wg pozycji.
    """

    # Filtrowanie danych do danego dnia
    data_day = data_plot.loc[data_plot.index.date == pd.to_datetime(date_plot).date()].copy()
    data_day = data_day[[col_signal, col_upper_entry, col_upper_exit, col_lower_entry, col_lower_exit, col_pos]].copy()

    # Przygotowanie kolumn czasowych
    data_day = data_day.reset_index()
    data_day.rename(columns={data_day.columns[0]: 'Time'}, inplace=True)
    data_day['next_Time'] = data_day['Time'].shift(-1)
    data_day['position'] = data_day[col_pos]

    # Kolory pozycji
    pos_colors = {
        -1: 'red',
         0: 'gray',
         1: 'green'
    }

    # Inicjalizacja wykresu
    fig, ax = plt.subplots(figsize=(width, height))

    # Rysowanie linii
    ax.plot(data_day['Time'], data_day[col_signal], color='black', label='Sygnał')

    ax.plot(data_day['Time'], data_day[col_upper_entry], color='blue', linestyle='--', linewidth=1.5, label='Upper Entry')
    ax.plot(data_day['Time'], data_day[col_upper_exit], color='blue', linestyle='-', linewidth=1.5, label='Upper Exit')
    ax.plot(data_day['Time'], data_day[col_lower_entry], color='darkorange', linestyle='--', linewidth=1.5, label='Lower Entry')
    ax.plot(data_day['Time'], data_day[col_lower_exit], color='darkorange', linestyle='-', linewidth=1.5, label='Lower Exit')

    # Pobranie zakresu osi Y
    ymin, ymax = ax.get_ylim()

    # Kolorowe tło wg pozycji
    for _, row in data_day.iterrows():
        if pd.isna(row['position']) or pd.isna(row['next_Time']):
            continue
        color = pos_colors.get(int(row['position']), 'white')
        ax.add_patch(Rectangle((row['Time'], ymin),
                               row['next_Time'] - row['Time'],
                               ymax - ymin,
                               facecolor=color,
                               alpha=0.2))

    # Formatowanie
    ax.set_title(title)
    ax.set_xlabel("Czas")
    ax.set_ylabel("Sygnał / Pozycja")

    # Legenda bez duplikatów
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper left')

    # Formatowanie osi X – HH:MM
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    start = data_day["Time"].iloc[0]
    end = data_day["Time"].iloc[-1]
    tick_locs = pd.date_range(start=start, end=end, freq='15min')
    ax.set_xticks(tick_locs)
    ax.set_xlim(start, end)
    plt.xticks(rotation=45)

    # Zapis wykresu jeśli potrzebny
    if save_graph:
        if file_name is None:
            raise ValueError("Musisz podać file_name jeśli save_graph=True")
        plt.savefig(file_name, bbox_inches='tight', dpi=300)

    plt.tight_layout()
    plt.show()
    return ax

