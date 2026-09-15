# Neuronska mreža od nule u Scali

Implementacija neuronske mreže sa jednim skrivenim slojem u programskom jeziku **Scala**, bez ijedne gotove biblioteke za mašinsko učenje. Sva logika — propagacija unapred, funkcija greške, gradijentni spust i propagacija greške unazad (*backpropagation*) — napisana je ručno.

Seminarski rad iz predmeta *Logičko i funkcijsko programiranje*.

## Šta radi

Ista mreža demonstrira se na dva problema:

- **XOR** — klasičan problem koji jedan neuron ne može da reši (mreža 2–4–1).
- **Iris** — prepoznavanje vrste cveta na osnovu 4 merenja (mreža 4–8–3), uz merenje tačnosti na test skupu.

Na kraju se pad greške kroz epohe iscrtava i snima kao `greska.png`.

## Fajlovi

- `NeuronskaMreza.scala` — kompletan izvorni kod
- `iris.csv` — Iris skup podataka (150 primera)
- `greska.png` — grafik pada greške (nastaje pokretanjem programa)

## Pokretanje

Potreban je [scala-cli](https://scala-cli.virtuslab.org). Iz foldera sa fajlovima pokrenuti:

```
scala-cli run NeuronskaMreza.scala
```

Direktiva na vrhu koda automatski preuzima biblioteku JFreeChart (za grafik), pa nije potrebna dodatna instalacija.

## Kako radi (ukratko)

Mreža uči iz primera ponavljajući ciklus:

1. napravi pogodak (*forward pass*),
2. izmeri grešku,
3. izračuna koliko je svaka težina kriva za grešku (*backpropagation*),
4. pomeri težine tako da greška bude manja (*gradient descent*).

Ponavljanjem kroz mnogo epoha, nasumične početne težine se postepeno doteruju do smislenih vrednosti.

## Autor

[ Ime i prezime, broj indeksa ]
