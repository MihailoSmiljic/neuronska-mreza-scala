//> using dep "org.jfree:jfreechart:1.5.4"

// =============================================================================
//  NEURONSKA MREZA "OD NULE" U SCALI
//  -----------------------------------------------------------------------------
//  Demonstriramo istu mrezu na dva problema:
//    1) XOR   -- "hello world" neuronskih mreza; jedan neuron ga ne moze resiti,
//                pa nam bas zato treba skriveni sloj.        (mreza 2 -> 4 -> 1)
//    2) Iris  -- prepoznavanje vrste cveta iz 4 merenja.     (mreza 4 -> 8 -> 3)
//
//  Na kraju se pad greske kroz epohe (za Iris) iscrtava i cuva u greska.png,
//  koristeci biblioteku JFreeChart.
//
//  Pokretanje (najlakse preko scala-cli, iz foldera gde je i iris.csv):
//      scala-cli run NeuronskaMreza.scala
// =============================================================================

import scala.util.Random
import scala.io.Source
import java.io.File
import org.jfree.chart.{ChartFactory, ChartUtils}
import org.jfree.data.xy.{XYSeries, XYSeriesCollection}

// -----------------------------------------------------------------------------
//  MREZA sa jednim skrivenim slojem.
//  Radi za bilo koje velicine slojeva, pa je koristimo i za XOR i za Iris.
// -----------------------------------------------------------------------------
class Mreza(brUlaza: Int, brSkrivenih: Int, brIzlaza: Int, seme: Int = 42) {

  private val rng = new Random(seme)
  private def nasumicno(): Double = rng.nextDouble() * 2 - 1  // broj iz [-1, 1]

  // Tezine i biasi pocinju NASUMICNO -- mreza na startu ne zna nista.
  //   w1(j)(i) = tezina od ulaza i ka skrivenom neuronu j
  //   w2(k)(j) = tezina od skrivenog neurona j ka izlaznom neuronu k
  private val w1 = Array.fill(brSkrivenih, brUlaza)(nasumicno())
  private val b1 = Array.fill(brSkrivenih)(nasumicno())
  private val w2 = Array.fill(brIzlaza, brSkrivenih)(nasumicno())
  private val b2 = Array.fill(brIzlaza)(nasumicno())

  // Aktivaciona funkcija i njen nagib (nagib treba za backpropagation).
  private def sigmoida(x: Double): Double = 1.0 / (1.0 + math.exp(-x))
  private def sigmoidaNagib(y: Double): Double = y * (1.0 - y)

  // FORWARD PASS: provuce ulaz kroz mrezu -> vrati (skriveni sloj, izlaz).
  def naprednaPropagacija(ulaz: Array[Double]): (Array[Double], Array[Double]) = {
    val skriveni = Array.tabulate(brSkrivenih) { j =>
      var s = b1(j)
      for (i <- 0 until brUlaza) s += w1(j)(i) * ulaz(i)
      sigmoida(s)
    }
    val izlaz = Array.tabulate(brIzlaza) { k =>
      var s = b2(k)
      for (j <- 0 until brSkrivenih) s += w2(k)(j) * skriveni(j)  // ulazi = izlazi skrivenog sloja
      sigmoida(s)
    }
    (skriveni, izlaz)
  }

  // Samo pogodak (bez skrivenog sloja).
  def predvidi(ulaz: Array[Double]): Array[Double] = naprednaPropagacija(ulaz)._2

  // JEDAN KORAK UCENJA na jednom primeru: forward -> greska -> backprop -> update.
  def trenirajJedan(ulaz: Array[Double], cilj: Array[Double], lr: Double): Double = {
    // 1) forward pass
    val (skriveni, izlaz) = naprednaPropagacija(ulaz)

    // 2) greska = suma kvadriranih razlika (pogodak - tacno)^2
    var greska = 0.0
    for (k <- 0 until brIzlaza) {
      val r = izlaz(k) - cilj(k)
      greska += r * r
    }

    // 3) backpropagation -- "delta" = koliko je neuron kriv za gresku
    val deltaIzlaz = Array.tabulate(brIzlaza) { k =>
      (izlaz(k) - cilj(k)) * sigmoidaNagib(izlaz(k))
    }
    val deltaSkriveni = Array.tabulate(brSkrivenih) { j =>
      var s = 0.0
      for (k <- 0 until brIzlaza) s += w2(k)(j) * deltaIzlaz(k)  // chain rule: prenesi krivicu unazad
      s * sigmoidaNagib(skriveni(j))
    }

    // 4) gradient descent -- pomeri tezine suprotno od nagiba
    for (k <- 0 until brIzlaza) {
      for (j <- 0 until brSkrivenih) w2(k)(j) -= lr * deltaIzlaz(k) * skriveni(j)
      b2(k) -= lr * deltaIzlaz(k)
    }
    for (j <- 0 until brSkrivenih) {
      for (i <- 0 until brUlaza) w1(j)(i) -= lr * deltaSkriveni(j) * ulaz(i)
      b1(j) -= lr * deltaSkriveni(j)
    }

    greska
  }

  // Trenira kroz vise epoha nad celim skupom; vraca prosecnu gresku po epohi.
  def treniraj(podaci: Array[(Array[Double], Array[Double])], epoha: Int, lr: Double): Array[Double] = {
    val istorija = Array.fill(epoha)(0.0)
    for (e <- 0 until epoha) {
      var ukupno = 0.0
      for ((ulaz, cilj) <- podaci) ukupno += trenirajJedan(ulaz, cilj, lr)
      istorija(e) = ukupno / podaci.length
    }
    istorija
  }
}

// -----------------------------------------------------------------------------
//  POMOCNE FUNKCIJE za Iris skup podataka.
// -----------------------------------------------------------------------------
object Podaci {
  val vrste = Array("setosa", "versicolor", "virginica")

  // Ucitaj iris.csv (zaglavlje + 150 redova: 4 broja + naziv vrste).
  def ucitajIris(putanja: String): Array[(Array[Double], String)] = {
    val izvor = Source.fromFile(putanja)
    val redovi = izvor.getLines().toArray
    izvor.close()
    redovi.drop(1).filter(_.trim.nonEmpty).map { red =>
      val polja = red.split(",")
      (polja.take(4).map(_.toDouble), polja(4).trim)
    }
  }

  // Min-max normalizacija svake kolone na opseg [0, 1] (da ucenje bude stabilno).
  def normalizuj(x: Array[Array[Double]]): Array[Array[Double]] = {
    val brKolona = x(0).length
    val min = Array.tabulate(brKolona)(c => x.map(_(c)).min)
    val max = Array.tabulate(brKolona)(c => x.map(_(c)).max)
    x.map { red =>
      Array.tabulate(brKolona) { c =>
        if (max(c) == min(c)) 0.0 else (red(c) - min(c)) / (max(c) - min(c))
      }
    }
  }

  // Naziv vrste -> one-hot vektor, npr. setosa -> [1, 0, 0].
  def oneHot(vrsta: String): Array[Double] = vrste.map(v => if (v == vrsta) 1.0 else 0.0)

  // Indeks najveceg izlaza = predvidjena klasa.
  def argmax(a: Array[Double]): Int = a.indices.maxBy(i => a(i))
}

// -----------------------------------------------------------------------------
//  GLAVNI PROGRAM
// -----------------------------------------------------------------------------
object Main {
  def main(args: Array[String]): Unit = {

    // =============================== 1) XOR ===============================
    println("=== XOR ===")
    val xorPodaci = Array(
      (Array(0.0, 0.0), Array(0.0)),
      (Array(0.0, 1.0), Array(1.0)),
      (Array(1.0, 0.0), Array(1.0)),
      (Array(1.0, 1.0), Array(0.0))
    )
    val xorMreza = new Mreza(2, 4, 1)
    xorMreza.treniraj(xorPodaci, epoha = 10000, lr = 0.5)
    for ((ulaz, cilj) <- xorPodaci) {
      val p = xorMreza.predvidi(ulaz)(0)
      println(f"ulaz [${ulaz.mkString(", ")}] -> pogodak $p%.3f   (tacno: ${cilj(0).toInt})")
    }

    // =============================== 2) IRIS ==============================
    println("\n=== IRIS ===")
    val sirovi = Podaci.ucitajIris("iris.csv")
    val obelezja = Podaci.normalizuj(sirovi.map(_._1))
    val ciljevi = sirovi.map(r => Podaci.oneHot(r._2))

    // izmesaj i podeli: 80% trening, 20% test
    val skup = new Random(1).shuffle(obelezja.zip(ciljevi).toList).toArray
    val granica = (skup.length * 0.8).toInt
    val (trening, test) = skup.splitAt(granica)

    val irisMreza = new Mreza(4, 8, 3)
    val istorija = irisMreza.treniraj(trening, epoha = 2000, lr = 0.3)

    // tacnost na test skupu
    val tacnih = test.count { case (ulaz, cilj) =>
      Podaci.argmax(irisMreza.predvidi(ulaz)) == Podaci.argmax(cilj)
    }
    val tacnost = tacnih.toDouble / test.length * 100
    println(f"Tacnost na test skupu: $tacnost%.1f%%  ($tacnih od ${test.length})")

    // ==================== GRAFIK PADA GRESKE (JFreeChart) ==================
    val serija = new XYSeries("Prosecna greska")
    istorija.zipWithIndex.foreach { case (g, e) => serija.add(e + 1, g) }
    val grafik = ChartFactory.createXYLineChart(
      "Pad greske kroz epohe (Iris)", "Epoha", "Prosecna greska",
      new XYSeriesCollection(serija)
    )
    ChartUtils.saveChartAsPNG(new File("greska.png"), grafik, 800, 600)
    println("Grafik sacuvan u greska.png")
  }
}
