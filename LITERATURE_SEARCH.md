# Wyszukiwanie literatury Raman/SERS

Uruchamiaj z katalogu głównego projektu (również w terminalu PyCharm):

```bash
python -m pip install -r requirements-literature.txt
python scripts/07_search_literature.py --dry-run
python scripts/07_search_literature.py --email TWOJ_EMAIL --limit 3
python scripts/07_search_literature.py --email TWOJ_EMAIL
```

Skrypt czyta `scripts/ActiveSubstanceExport_27-02-2026.xlsx`, wyszukuje nazwy i numery CAS ze słowami Raman, SERS oraz surface enhanced Raman. Nie filtruje statusu regulacyjnego ani lat publikacji. Nie wymaga ponownego importu EU ani zmian konfiguracji DFT. Inny plik wybierz przez `--xlsx SCIEZKA`.

OpenAlex: opcjonalnie ustaw `OPENALEX_API_KEY` w zmiennych środowiskowych konfiguracji Run w PyCharm. Kontakt Crossref można ustawić przez `CROSSREF_MAILTO`. Nie zapisuj klucza w repozytorium. Dokumentacja: https://help.openalex.org/api/authentication/ oraz https://www.crossref.org/documentation/retrieve-metadata/rest-api/.

Domyślnie pobierana jest pierwsza strona (100 wyników) każdego zapytania od każdego dostawcy. `--pages 3` rozszerza wyszukiwanie do 3 stron; kolumna `capped` w Query_log wskazuje niewyczerpane wyniki. To wyszukiwanie kandydatów, nie systematyczny pełny przegląd. Crossref zwraca wyniki według podobieństwa, także niepowiązane artykuły. Synonimy inne niż nazwa z Excela nie są automatycznie rozwijane. Skrót SERS ani słowo Raman nie potwierdzają techniki w konkretnym eksperymencie.

Wyniki w `data/literature/`:

- `literature.sqlite`: surowe metadane, powiązania substancji z publikacjami i dziennik zapytań; niezależna od bazy DFT.
- `literature_candidates.xlsx`: Candidates (DOI, linki OA, metadane licencji), Substances (wszystkie związki) i Query_log (postęp, błędy, ograniczenia).

Uruchomienie tego samego polecenia pomija zakończone zapytania i ponawia błędne. Zwiększenie `--pages` powoduje nowe zapytania. Aby wykonać świeży przegląd z tą samą liczbą stron, użyj nowego `--out data/literature_new`. Błąd dostawcy przerywa przebieg po zapisaniu postępu; ponów później lub wybierz `--providers crossref`. Eksport bez sieci: `python scripts/07_search_literature.py --export-only`.

Excel jest nadpisywany przy eksporcie. Do ręcznej weryfikacji zachowaj kopię pod inną nazwą. Wszystkie rekordy mają początkowo UNVERIFIED: sprawdź obecność eksperymentalnego widma, technikę (Raman/SERS/oba), numer rysunku, supplementary, warunki pomiarowe i możliwość ponownego wykorzystania. Metadane licencji nie stanowią potwierdzenia prawa do publikowania odtworzonych widm. Link OA nie jest dowodem dostępności danych liczbowych. Moduł nie pobiera PDF ani nie digitalizuje wykresów.

Testy offline: `python -m unittest discover -s tests -p 'test_literature_search.py'`.
