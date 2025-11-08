# MER

## Autorzy
- Marta Sobol
- Maciej Kozłowski
- Adam Dąbkowski

## Analiza literatury

|Publikacja|Link|Komentarz|Kod|Wytrenowane modele|Metryki|Zasoby obliczeniowe|
|-|-|-|-|-|-|-|
|**R. Liyanarachchi, A. Joshi, E. Meijering, 2025** *“A Survey on Multimodal Music Emotion Recognition.”*|[🔗](https://arxiv.org/abs/2504.18799)|Dziedzina MER przesuwa się ku multimodalności, łącząc audio, tekst, wideo, sygnały fizjologiczne, dane symboliczne i metadane. Praca porządkuje proces MMER w czterech etapach (dobór danych i modalności, ekstrakcja cech, fuzja i przetwarzanie, predykcja) oraz omawia fuzję wczesną, pośrednią, późną oraz mechanizmy uwagi i pamięci. Przegląd dotychczasowej literatury prezentuje przejście od klasycznych modeli i SVM do CNN/LSTM, a następnie do transformerów i metod łączących modalności. Większość publicznie dostępnych zbiorów ma charakter jednomodalny. Brakuje spójnych benchmarków i ujednoliconych metryk, co ogranicza porównywalność wyników. W kontekście wykorzystania krytyczne pozostają zasoby obliczeniowe i synchronizacja strumieni, natomiast rekomendowane są lżejsze mechanizmy fuzji oraz uczenie transferowe.|❌|❌|Accuracy <br> Precision <br> Recall <br> F1 <br> AUROC <br> MAE <br> RMSE <br> R² <br> CCC <br> hits@k <br> MAP@k|❌|
|**J. Kang, D. Herremans, 2024** *“Are We There Yet? A Brief Survey of Music Emotion Prediction Datasets, Models and Outstanding Challenges”*|[🔗](https://arxiv.org/pdf/2406.08809)|Porównanie istniejących zbiorów danych i architektur modeli. Najpopularniejszym dostępnym zbiorem jest **DEAM**, który zawiera ponad 2000 utworów muzycznych (głównie rock i muzyka elektroniczna) w formacie MP3 o długości 45 sekund sparowanych z wartościami **arousal** i **valence** - dwuwymiarowy model Russell'a reprezentujący emocje. Jest on powszechnie wykorzystywany w MER do adnotowania. Coraz częściej można spotkać się z podejściami wielomodalnymi w celu poprawy jakości predykcji - związane jest to z różnymi źródłami bodźców sensorycznych u człowieka, np. modele wykorzystujące poza samym dźwiękiem także wideo.|❌|❌|Accuracy <br> F1 <br> PR-AUC <br> ROC-AUC <br> RMSE <br> R² <br> CCC <br> Pearson correlation |❌|
|**Pedro Lima Louro, Hugo Redinho, Ricardo Malheiro, Rui Pedro Paiva, Renato Panda, 2024** *“A Comparison Study of Deep Learning Methodologies for Music Emotion Recognition.”*|[🔗](https://www.mdpi.com/1424-8220/24/7/2201)|Artykuł porównuje klasyczne metody uczenia maszynowego i metody uczenia głębokiego w zadaniu klasyfikacji emocji 4Q. Autorzy przeprowadzili eksperymenty z różnymi architekturami modeli, technikami augmetacji danych, sposobami reprezentacji danych oraz uczeniem transferowym. Najlepsze wyniki uzyskano przy zastosowaniu podejścia hybrydowego, łączacego CNN trenowanego na rozszerzonym zbiorze danych i DNN wykorzystującego mel-spektrogramy oraz ręcznie wyekstrahowane cechy. Ten model osiągnął 80,2% F1-score, co stanowiło znaczną poprawę w porównaniu do najlepszych modeli bazowych. Ponadto pokazano, że zwiększenie ilości danych miało większy wpływ niż równoważenie klas, a klasyczne techniki augmentacji poprawiały skuteczność modeli. Natomiast zastosowanie architektur działających na poziomie segmentów (segment-level), uczenia transferowego lub embeddingów, nie przyniosło poprawy wyników - były one gorsze od modeli bazowych.|❌|❌|Precision <br> Recall <br> F1|Eksperymenty były przeprowadzane na współdzielonym serwerze z dwoma procesorami Intel Xeon Silver 4214 (48 rdzeni, 2,2 GHz) oraz trzema kartami NVIDIA Quadro P500 (16 GB), a w razie potrzeby korzystano także z Google Colab z kartami NVIDIA P100 lub T4.|
|**Pedro Lima Louro, Hugo Redinho, Ricardo Santos, Ricardo Malheiro, Renato Panda, Rui Pedro Paiva, 2025** *“MERGE — A Bimodal Audio-Lyrics Dataset for Static Music Emotion Recognition”*|[🔗](https://arxiv.org/abs/2407.06060)|Artykuł stanowi odpowiedź na brak publicznych, dużych i kontrolowanych jakościowo zbiorów bimodalnych audio+tekst dla MER. Autorzy przedstawiają trzy nowe zbiory: MERGE Audio, MERGE Lyrics oraz MERGE Bimodal, etykietowane w czterech ćwiartkach Russella (valence–arousal). Dane powstały półautomatycznie na bazie metadanych i klipów z bazy AllMusic, z kontrolą jakości i standaryzacją próbek. |❌|❌|F1 <br> RMSE <br> R²|❌|
|**Essentia**|[🔗](https://essentia.upf.edu/models.html)|Serwis udostępnia pre-trenowane modele do analizy muzyki wraz z wagami, metadanymi i przykładami użycia.|✔️|✔️|Metryki są zróżnicowane w zależności od rozpatrywanego modelu|❌|


## Status realizacji

✔️ Wykonano

- Analiza wymagań i literatury z zakresu MER.
- Analiza wybranych zbiorów danych (DEAM, emoMusic, MERGE) i przygotowanie środowiska.
- Implementacja prototypu bazowego na danych pozbawionych dodatkowego przeprocesowania (czyszczenia i augmentacji). Prototyp umożliwia wczytanie pliku audio, jego analizę i zwrócenie predykcji (tryb VA).
- Porównanie wyników otrzymanego prototypu z modelami dostępnymi w Essentia (tryb VA).

🚧 W trakcie realizacji

- Dostosowanie struktury repozytorium do szablonu *cookiecutter-data-science*
- Integracja z tensorboard
- Eksperymenty z różnymi architekturami modeli oraz analiza wpływu augmentacji danych
- Opracowanie aplikacji webowej

## Do pobrania
### DEAM dataset
audio - https://cvml.unige.ch/databases/DEAM/DEAM_audio.zip do katalogu `/data/DEAM/audio/`

annotations - https://cvml.unige.ch/databases/DEAM/DEAM_Annotations.zip do katalogu `/data/DEAM/annotations/`
