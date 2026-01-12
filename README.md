# MER

## Autorzy
- Marta Sobol
- Maciej Kozłowski
- Adam Dąbkowski

## Uruchomienie

```bash
# Klonowanie repozytorium
git clone [<repository_url>](https://github.com/AdamixD/WIMU-Z4.git)
cd WIMU-Z4

# Utworzenie środowiska wirtualnego
python -m venv venv
source venv/bin/activate  # Linux/macOS
# lub: venv\Scripts\activate  # Windows

# Instalacja zależności
pip install -r requirements.txt
```
Lista dostępnych komend znajduje się w pliku COMMANDS.md

## Analiza literatury

|Publikacja|Link|Komentarz|Kod|Wytrenowane modele|Metryki|Zasoby obliczeniowe|
|-|-|-|-|-|-|-|
|**R. Liyanarachchi, A. Joshi, E. Meijering, 2025** *“A Survey on Multimodal Music Emotion Recognition.”*|[🔗](https://arxiv.org/abs/2504.18799)|Dziedzina MER przesuwa się ku multimodalności, łącząc audio, tekst, wideo, sygnały fizjologiczne, dane symboliczne i metadane. Praca porządkuje proces MMER w czterech etapach (dobór danych i modalności, ekstrakcja cech, fuzja i przetwarzanie, predykcja) oraz omawia fuzję wczesną, pośrednią, późną oraz mechanizmy uwagi i pamięci. Przegląd dotychczasowej literatury prezentuje przejście od klasycznych modeli i SVM do CNN/LSTM, a następnie do transformerów i metod łączących modalności. Większość publicznie dostępnych zbiorów ma charakter jednomodalny. Brakuje spójnych benchmarków i ujednoliconych metryk, co ogranicza porównywalność wyników. W kontekście wykorzystania krytyczne pozostają zasoby obliczeniowe i synchronizacja strumieni, natomiast rekomendowane są lżejsze mechanizmy fuzji oraz uczenie transferowe.|❌|❌|Accuracy <br> Precision <br> Recall <br> F1 <br> AUROC <br> MAE <br> RMSE <br> R² <br> CCC <br> hits@k <br> MAP@k|❌|
|**J. Kang, D. Herremans, 2024** *“Are We There Yet? A Brief Survey of Music Emotion Prediction Datasets, Models and Outstanding Challenges”*|[🔗](https://arxiv.org/pdf/2406.08809)|Porównanie istniejących zbiorów danych i architektur modeli. Najpopularniejszym dostępnym zbiorem jest **DEAM**, który zawiera ponad 2000 utworów muzycznych (głównie rock i muzyka elektroniczna) w formacie MP3 o długości 45 sekund sparowanych z wartościami **arousal** i **valence** - dwuwymiarowy model Russell'a reprezentujący emocje. Jest on powszechnie wykorzystywany w MER do adnotowania. Coraz częściej można spotkać się z podejściami wielomodalnymi w celu poprawy jakości predykcji - związane jest to z różnymi źródłami bodźców sensorycznych u człowieka, np. modele wykorzystujące poza samym dźwiękiem także wideo.|❌|❌|Accuracy <br> F1 <br> PR-AUC <br> ROC-AUC <br> RMSE <br> R² <br> CCC <br> Pearson correlation |❌|
|**Pedro Lima Louro, Hugo Redinho, Ricardo Malheiro, Rui Pedro Paiva, Renato Panda, 2024** *“A Comparison Study of Deep Learning Methodologies for Music Emotion Recognition.”*|[🔗](https://www.mdpi.com/1424-8220/24/7/2201)|Artykuł porównuje klasyczne metody uczenia maszynowego i metody uczenia głębokiego w zadaniu klasyfikacji emocji 4Q. Autorzy przeprowadzili eksperymenty z różnymi architekturami modeli, technikami augmetacji danych, sposobami reprezentacji danych oraz uczeniem transferowym. Najlepsze wyniki uzyskano przy zastosowaniu podejścia hybrydowego, łączacego CNN trenowanego na rozszerzonym zbiorze danych i DNN wykorzystującego mel-spektrogramy oraz ręcznie wyekstrahowane cechy. Ten model osiągnął 80,2% F1-score, co stanowiło znaczną poprawę w porównaniu do najlepszych modeli bazowych. Ponadto pokazano, że zwiększenie ilości danych miało większy wpływ niż równoważenie klas, a klasyczne techniki augmentacji poprawiały skuteczność modeli. Natomiast zastosowanie architektur działających na poziomie segmentów (segment-level), uczenia transferowego lub embeddingów, nie przyniosło poprawy wyników - były one gorsze od modeli bazowych.|❌|❌|Precision <br> Recall <br> F1|Eksperymenty były przeprowadzane na współdzielonym serwerze z dwoma procesorami Intel Xeon Silver 4214 (48 rdzeni, 2,2 GHz) oraz trzema kartami NVIDIA Quadro P500 (16 GB), a w razie potrzeby korzystano także z Google Colab z kartami NVIDIA P100 lub T4.|
|**Pedro Lima Louro, Hugo Redinho, Ricardo Santos, Ricardo Malheiro, Renato Panda, Rui Pedro Paiva, 2025** *“MERGE — A Bimodal Audio-Lyrics Dataset for Static Music Emotion Recognition”*|[🔗](https://arxiv.org/abs/2407.06060)|Artykuł stanowi odpowiedź na brak publicznych, dużych i kontrolowanych jakościowo zbiorów bimodalnych audio+tekst dla MER. Autorzy przedstawiają trzy nowe zbiory: MERGE Audio, MERGE Lyrics oraz MERGE Bimodal, etykietowane w czterech ćwiartkach Russella (valence–arousal). Dane powstały półautomatycznie na bazie metadanych i klipów z bazy AllMusic, z kontrolą jakości i standaryzacją próbek. |❌|❌|F1 <br> RMSE <br> R²|❌|
|**Essentia**|[🔗](https://essentia.upf.edu/models.html)|Serwis udostępnia pre-trenowane modele do analizy muzyki wraz z wagami, metadanymi i przykładami użycia.|✔️|✔️|Metryki są zróżnicowane w zależności od rozpatrywanego modelu|❌|

## Zbiory danych

**DEAM**

| Parametr             | Wartość                  |
|----------------------|--------------------------|
| Liczba utworów       | 1802                     |
| Typ adnotacji        | Dynamiczne (per sekunda) |
| Reprezentacja emocji | VA                       |

**PMEmo**

| Parametr             | Wartość                  |
|----------------------|--------------------------|
| Liczba utworów       | 767                      |
| Typ adnotacji        | Dynamiczne (per sekunda) |
| Reprezentacja emocji | VA                       |

**MERGE**

| Parametr              | Wartość                |
|-----------------------|------------------------|
| Liczba utworów        | 3554                   |
| Typ adnotacji         | Statyczne (cały utwór) |
| Predefiniowane splity | 70/15/15 lub 40/30/30  |
| Reprezentacja emocji  | VA lub Russell4Q       |


## Eksperymenty

### Metryki ewaluacji
W eksperymentach wykorzystano następujące metryki:
- **CCC (Concordance Correlation Coefficient)** - dla trybu VA, mierzy zgodność między predykcjami a wartościami rzeczywistymi, uwzględniając zarówno korelację jak i średnie wartości
- **F1 Score (weighted)** - dla trybu Russell4Q, harmoniczna średnia precyzji i recall, ważona rozmiarem klas
  
### Metodologia eksperymentów
Każdy eksperyment składał się z dwóch faz:

**Faza 1: Optymalizacja hiperparametrów**
- 10 triali Optuna z algorytmem TPE
- Walidacja k-fold (k=5) dla DEAM i PMEmo
- Predefiniowane splity train/valid/test (70/15/15) dla MERGE
- Metryka optymalizacji: CCC_mean (VA) lub F1 (Russell4Q)

**Faza 2: Trening finalnego modelu**
- Wykorzystanie najlepszych znalezionych hiperparametrów
- Trening na pełnym zbiorze treningowym
- Ewaluacja na zbiorze testowym

### Wyniki eksperymentów
Otrzymane wyniki eksperymentów na zbiorze testowym dla najlepszego modelu

**Tryb VA**

| Zbiór danych | Głowa BiGRU   | Głowa CNNLSTM |
|--------------|---------------|---------------|
| DEAM         | 0.637         | 0.725         |
| PMEmo        | 0.646         | 0.710         |
| Merge        | 0.470         | 0.427         |

**Tryb Russell4Q**

W tym trybie etykiety VA dla zbiorów DEAM i PMEmo są mapowane do kwadrantów modelu Russella.

| Zbiór danych | Głowa BiGRU | Głowa CNNLSTM |
|--------------|-------------|---------------|
| DEAM         | 0.623       | 0.698         |
| PMEmo        | 0.670       | 0.734         |
| Merge        | 0.548       | 0.529         |

### Augmentacje
- shift – przesunięcie czasowe sygnału.
- gain – zmiana głośności nagrania.
- reverb – dodanie pogłosu do sygnału.
- lowpass – zastosowanie filtru dolnoprzepustowego.
- highpass – zastosowanie filtru górnoprzepustowego.
- bandpass – filtr pasmowy przepuszczający wybrane częstotliwości.
- pitch_shift – zmiana wysokości tonu nagrania.

### Wyniki
Otrzymane wyniki na zbiorze testowym uzyskano przy treningu, w którym dla każdej augmentacji 30% oryginalnych danych było przetwarzanych w formie augmentowanej i dodawanych do zbioru treningowego.

**PMEmo**
| Tryb / Model | BiGRU | CNNLSTM |
|--------------|-------|----------|
| VA           | 0.7160 | 0.7638 |
| Russell4Q    | 0.7434 | 0.8012 |

**Merge**
| Tryb / Model | BiGRU | CNNLSTM |
|--------------|-------|----------|
| VA           | 0.4879 | 0.4779 |
| Russell4Q    | 0.5614 | 0.5399 |


### Wnioski
**Porównanie głów**

Głowa CNNLSTM osiąga zauważalnie lepsze wyniki niż BiGRU (przewaga 10-14%) na zbiorach DEAM i PMEmo w obu trybach, natomiast dla zbioru Merge lepsze rezultaty uzyskuje BiGRU. Wskazuje to, że w przypadku danych dynamicznych skuteczniejsza jest architektura CNNLSTM, która umożliwia lepsze modelowanie zależności czasowych. Z kolei dla danych statycznych korzystniejsza okazuje się prostsza architektura BiGRU, charakteryzująca się lepszą zdolnością do generalizacji.

**Porównanie zbiorów danych**

Dla zbioru Merge uzyskane wyniki są wyraźnie niższe, niezależnie od zastosowanego trybu, co wskazuje, że jest on najbardziej wymagającym z analizowanych zbiorów danych. Sugeruje to, że statyczne adnotacje emocji stanowią większe wyzwanie dla zastosowanych modeli, które znacznie lepiej radzą sobie z adnotacjami dynamicznymi. Prawdopodobnie wynika to z faktu, że statyczne etykiety, przypisane do całego utworu, nie pozwalają w pełni wykorzystać potencjału architektur sekwencyjnych, zaprojektowanych do modelowania zależności czasowych.

Najwyższe wyniki uzyskano dla zbioru PMEmo, jednak różnice w porównaniu do zbioru DEAM są stosunkowo niewielkie. Może to wskazywać, że oba zbiory charakteryzują się podobnym poziomem trudności oraz spójnością adnotacji, a zastosowane modele efektywnie wykorzystują dynamiczną reprezentację emocji w obu przypadkach.

**Porównanie trybów**

Największe różnice między trybami VA i Russell4Q widoczne są dla zbioru Merge, gdzie lepsze wyniki uzyskano w trybie Russell4Q. Dla PMEmo Russell4Q również jest nieznacznie lepszy. Jedynie w zbiorze DEAM tryb VA daje nieco lepsze rezultaty.

W zbiorach DEAM i PMEmo wartości VA zostały mapowane na kwadranty Russella, mimo to dyskretna reprezentacja zachowuje istotne informacje i pozwala modelom skutecznie uczyć się wzorców emocjonalnych.

**Augmentacje**

Dodanie augmentacji poprawia wyniki modeli, co jest szczególnie widoczne w przypadku zbioru PMEmo (poprawa o 7–11%). Może to wynikać z faktu, że jest to najmniejszy ze zbiorów (tylko 767 utworów), a wprowadzenie danych augmentowanych pozwoliło zwiększyć liczbę próbek treningowych. Dla zbioru Merge poprawa wyników jest natomiast jedynie nieznaczna, co prawdopodobnie wynika z jego dużej wielkości (3554 utworów). Wynika z tego, że stosowanie augmentacji jest szczególnie korzystne dla mniejszych zbiorów danych.

### Aplikacja webowa

**Funkcjonalności**

1. **Ładowanie modeli** - wybór z dostępnych modeli .pth
2. **Upload audio** - wgrywanie plików MP3/WAV
3. **Wizualizacja VA** - wykres valence/arousal w czasie
4. **Wizualizacja Russell4Q** - rozkład kwadrantów
5. **Porównanie modeli** - analiza dwóch modeli jednocześnie
6. **Odtwarzacz audio** - synchronizacja z wizualizacjami

**Interfejs**



