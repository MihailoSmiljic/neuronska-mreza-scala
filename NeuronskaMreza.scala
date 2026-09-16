//> using dep "org.jfree:jfreechart:1.5.4"

import scala.util.Random
import scala.io.Source
import java.io.File
import org.jfree.chart.{ChartFactory, ChartUtils}
import org.jfree.data.xy.{XYSeries, XYSeriesCollection}

class Mreza(brUlaza: Int, brSkrivenih: Int, brIzlaza: Int, seme: Int = 42) {

  private val rng = new Random(seme)
  private def nasumicno(): Double = rng.nextDouble() * 2 - 1  // broj iz [-1, 1]

  private val w1 = Array.fill(brSkrivenih, brUlaza)(nasumicno())
  private val b1 = Array.fill(brSkrivenih)(nasumicno())
  private val w2 = Array.fill(brIzlaza, brSkrivenih)(nasumicno())
  private val b2 = Array.fill(brIzlaza)(nasumicno())

  private def sigmoida(x: Double): Double = 1.0 / (1.0 + math.exp(-x))
  private def sigmoidaNagib(y: Double): Double = y * (1.0 - y)

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

  def predvidi(ulaz: Array[Double]): Array[Double] = naprednaPropagacija(ulaz)._2

  def trenirajJedan(ulaz: Array[Double], cilj: Array[Double], lr: Double): Double = {
    val (skriveni, izlaz) = naprednaPropagacija(ulaz)

    var greska = 0.0
    for (k <- 0 until brIzlaza) {
      val r = izlaz(k) - cilj(k)
      greska += r * r
    }

    val deltaIzlaz = Array.tabulate(brIzlaza) { k =>
      (izlaz(k) - cilj(k)) * sigmoidaNagib(izlaz(k))
    }
    val deltaSkriveni = Array.tabulate(brSkrivenih) { j =>
      var s = 0.0
      for (k <- 0 until brIzlaza) s += w2(k)(j) * deltaIzlaz(k)  // chain rule: prenesi krivicu unazad
      s * sigmoidaNagib(skriveni(j))
    }

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

object Podaci {
  val vrste = Array("setosa", "versicolor", "virginica")

  def ucitajIris(putanja: String): Array[(Array[Double], String)] = {
    val izvor = Source.fromFile(putanja)
    val redovi = izvor.getLines().toArray
    izvor.close()
    redovi.drop(1).filter(_.trim.nonEmpty).map { red =>
      val polja = red.split(",")
      (polja.take(4).map(_.toDouble), polja(4).trim)
    }
  }

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

  def oneHot(vrsta: String): Array[Double] = vrste.map(v => if (v == vrsta) 1.0 else 0.0)

  def argmax(a: Array[Double]): Int = a.indices.maxBy(i => a(i))
}

object Main {
  def main(args: Array[String]): Unit = {

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

    println("\n=== IRIS ===")
    val sirovi = Podaci.ucitajIris("iris.csv")
    val obelezja = Podaci.normalizuj(sirovi.map(_._1))
    val ciljevi = sirovi.map(r => Podaci.oneHot(r._2))

    val skup = new Random(1).shuffle(obelezja.zip(ciljevi).toList).toArray
    val granica = (skup.length * 0.8).toInt
    val (trening, test) = skup.splitAt(granica)

    val irisMreza = new Mreza(4, 8, 3)
    val istorija = irisMreza.treniraj(trening, epoha = 2000, lr = 0.3)

    val tacnih = test.count { case (ulaz, cilj) =>
      Podaci.argmax(irisMreza.predvidi(ulaz)) == Podaci.argmax(cilj)
    }
    val tacnost = tacnih.toDouble / test.length * 100
    println(f"Tacnost na test skupu: $tacnost%.1f%%  ($tacnih od ${test.length})")

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
