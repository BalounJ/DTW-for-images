# Information retrieval pomocí Dynamic Time Warping algoritmu pro obrázky textu
Cílem je otestovat možnost použití algoritmu DTW pro nalezení odpovídajících obrázků podle vzoru. 

## Dynamic Time Warping (DTW)
Algoritmus pro Dynamic Time Warping vychází z [1]. Výstupem algoritmu je hodnota představující odlišnost dvou signálů. Je připravena základní varianta a varianta s omezením výpočtu pro zvolené okno (viz [/examples/dtw_example.ipynb](https://github.com/BalounJ/DTW-for-images/blob/master/examples/dtw_example.ipynb)). Vzhledem k rozměrům obrázků nemůže dojít k přetečení (pro int32) a normalizace proto u DTW není řešena. Vzhledem k časové náročnosti je pro vyhodnocení MAP použita implementace z modulu cdtw.

## Princip řešení
Řešení spočívá v převodu obrázku na signál a následném porovnávání signálů. Na základě porovnávání je odpověď tvořena množinou obrázků, které jsou seřazeny podle podobnosti se vzorovým obrázkem. Výsledek může vypadat podobně jako v [/examples/QbE_example.ipynb](https://github.com/BalounJ/DTW-for-images/blob/master/examples/QbE_example.ipynb).

### Převod obrázku na signál
Signál je vytvořen jako obsah inkoustu ve sloupcích obrázku (viz [/examples/image_signal_example.ipynb](https://github.com/BalounJ/DTW-for-images/blob/master/examples/image_signal_example.ipynb)). Předtím je černobílý obrázek normalizován na hodnoty 0 (bílá) a 1 (černá).

## Vyhodnocení
Vyhodnocení (více viz [/examples/MAP_eval.ipynb](https://github.com/BalounJ/DTW-for-images/blob/master/examples/MAP_eval.ipynb)) probíhá na Parzival Database [1], která je zpracována do následující struktury:
```
data   
└─── slovo1
│     │   img011.png
│     │   img012.png
│     │   ...
│   
└─── slovo2
      │   img021.png
      │   img022.png
      │   ...
```

## Závěr
Dosažených 37,7 % MAP pro dotaz podle vzoru vzhledem k jednoduchosti řešení není až tak špatný výsledek. Lepších výsledků by mohlo být dosaženo použitím dalších signálů (např. horní obrys) a jejich dalším zpracováním. DTW je tedy možné použít pro hledání obrázků podle vzoru, ale existují i lepší řešení jako např. použití neuronových sítí, kde je možné dosáhnout přes 90 % MAP pro dotazy podle vzoru i řetězce. 

## Závislosti
* cdtw `pip install cdtw`
* matplotlib `pip install matplotlib`
* numpy `pip install numpy`
* opencv `pip install opencv-python`

[1]: Rath, T. M. – Manmatha, R. Word spotting for historical documents. 
International Journal of Document Analysis and Recognition (IJDAR). Apr
2007, 9, 2, s. 139–152. ISSN 1433-2825. doi: 10.1007/s10032-006-0027-8.

[2] Fischer, A. et al. Lexicon-free handwritten word spotting using character HMMs. Pattern Recognition Letters. 2012, 33, 7, s. 934 – 942. ISSN 0167-8655.
