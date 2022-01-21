# Névelem felismerés és anonimizálás a Digital Twin Distiller segítségével​

## Háttér

A [MONTANA Tudásmenedzsment Kft](https://montana.hu/) által fejlesztett
[digital-twin-distiller](https://github.com/montana-knowledge-management/digital-twin-distiller)
egy szövegbányászatot és numerikus modellezési feladatok fejlesztési folyamatát támogató,
**python** nyelven implementált **számítási platform**, amely megkönnyíti többek között a mesterséges intelligencia
alapú, szövegbányászati, képfelismerési vagy egyéb komplex mérnöki problémák megoldása
során ismétlődő(rész)feladatok szervezését és menedzselését.

A **digitális iker** fogalmát eredetileg a gyártási folyamatok fejlesztése során alkották meg,
ez azonban az utóbbi években átalakult, kitágult, immár többféle definíciója is létezik.
Fő jelentésében az élő és élettelen entitások olyan digitális replikációját értjük alatta, ahol a termék teljes
életciklusa alatt keletkezett adatokat, numerikus szimulációkat, vagy akár a mesterséges
intelligencián alapuló nyelvi modelleket kezelhetjük.

A digital-twin-distiller esetében a **dockerizálásnak** köszönhetően az előállított digitális
ikrek később, akár évek múlva is futtathatók lesznek **a rendszer mélyebb ismerete nélkül**,
ilyen módon pedig segít kiküszöbölni a vállalati kompetencia fluktuációja, valamint a
vendor lock-in és a szoftverkörnyezet változása miatt esetlegesen bekövetkező, üzletmenetbeli
fennakadásokat.

A jelen projekt keretében elérhető névelem-felismerési feladaton alapuló anonimizáló a
digital-twin-distiller keretrendszer egy gyakorlati alkalmazása.


## Használat

Az API az adatot JSON formátumban várja, az elemzendő mondatot a "text" attribútum
értékeként megadva:

```
{
  "text": "Példa mondat."
}
```

## Alkalmazott modell

A keretrendszer képes bármilyen fájlba menthető illetve onnan beolvasható gépi
tanulási modellt kezelni egy projekten belül, tehát egyaránt alkalmas pl. `sklearn`,
`keras`, `huggingface` stb. alapú modellek hatékony használatára API-ként.

A jelen implementációban egy ... modell szerepel.

Ábra: F1, stb. (teljesítmény?)

## Kimeneti formátum

IOB2 taggelési formátum?

Elkerülve a monogramok használatát, a rendszer a neveket
„X” karakterre cseréli kizárólag az első karaktert meghagyva
és a rövidítést ponttal jelölve?



## Endpoint-ok

* `/` endpoint-on érhető el ez a dokumentáció.

* `/process` endpoint, ahol az alkalmazás `single input`-ként vájra a JSON-t amelyet címkézve visszaad.

* `/apidocs` endpoint, ahol az alkalmazás OpenAPI dokumentációként leírt működése, valamint teszt interfésze található.

* `/ping` endpoint tesztelhető a szerver elérése.

## Az integrált teszt interfész használata az `/apidocs` endpoint-on

![Használat](project/docs/docs/images/usage_.gif)
