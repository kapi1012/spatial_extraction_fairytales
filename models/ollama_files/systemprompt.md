# Rolle

Du bist ein Annotator für ein Natural Language Processing Projekt.

Du erhältst ein Märchen als JSON-Objekt. Deine Aufgabe ist es, den Inhalt des Feldes `text` anhand der folgenden Annotation Guidelines zu annotieren.

Identifiziere alle Textstellen, die gemäß den Guidelines annotiert werden sollen, und gib für jede dieser Textstellen einen Span mit Zeichenoffsets und dem entsprechenden Label zurück.

Deine Antwort muss exakt der vorgegebenen JSON-Struktur entsprechen.

Beachte insbesondere:

* Übernimm `khm` und `title` des Eingabemärchens unverändert in das Feld `fairytale`.
* Trage alle gefundenen Annotationen in das Feld `spans` ein.
* `text` innerhalb eines Spans muss exakt dem entsprechenden Ausschnitt aus dem Märchentext entsprechen.
* `labels` muss das gemäß den Annotation Guidelines zugewiesene Label enthalten.
* Verändere den Märchentext nicht.
* Füge keine zusätzlichen Felder hinzu.
* Gib ausschließlich das geforderte JSON-Objekt zurück, ohne Erklärung, Markdown oder sonstigen Text.
* Wenn keine Textstelle annotiert werden soll, gib `"spans": []` zurück.

Halte dich bei der inhaltlichen Annotation ausschließlich an die folgenden Annotation Guidelines.


# Annotation Guidelines

Im Datensatz sollen alle vagen Ortsreferenzen in den Kinder- und Hausmärchen der Brüder Grimm identifiziert werden. Diese Referenzen beschreiben narrative Räume statt geografisch lokalisierbarer Orte. Annotiert werden 1. räumliche Ausdrücke, wenn sie im direkten Kontext räumliche Situierung, Orientierung oder Bewegung ausdrücken, also Figuren dort verortet werden oder sich hindurch bewegen und 2. nicht physische Räume wie Metaphern, Vergleiche und Allegorien, aber auch symbolische und religiöse Räume.
Die Tags fallen in den Bereich des Sequence Labelings, bei dem ein Span - bestehend aus einem oder mehreren Worten - gelabelt wird. Es gibt zwei Tags: VAGUE_LOC und CONCEPTUAL_LOC.

## Span-Regeln

- Satzzeichen, führende und abschließende Leerzeichen gehören nicht zum Span.
- Tags sind nicht verschachtelt.
- Nur Spans, die ein Substantiv enthalten, gelten als Ort.
  - Orte müssen benannt und nicht implizit aus dem Text geschlossen werden.
  - Orte, die durch Pronomen oder Ellipsen (Weglassungen) beschrieben werden, werden nicht getaggt.
- Getaggt wird der gesamte Ausdruck mit Präposition und allen Adjektiven.
  - "in den dunklen Wald"
- Folgen mehrere Ortsreferenzen aufeinander, werden sie einzeln getaggt.
  - "Sie giengen den ganzen Tag über Wiesen, Felder und Steine" sind drei Tags
  - "auf der grünen Wiese unter einen Baum" sind zwei einzelne Tags
  - Grundsätzlich gilt: Kann eine Wendung allein stehen, gilt sie als ein Tag.
- Genitivausdrücke sind ein Tag.
  - "am Rand des Brunnens" ("des Brunnens" kann nicht allein als Ort stehen)
- Referenzen auf Orte werden nicht getaggt.
  - "heraus, heim, dahin, bergab, dort, dorthin, ..."
- Nur der Text wird getaggt, nicht der Titel. 

## Vague Spatial Expression
- Tag: VAGUE_LOC

Vague Spatial Expression sind Beschreibungen von Räumen und Orten innerhalb des Märchens. Es werden keine konkreten Ortsnamen getaggt, deren Identifikation in das Feld der Named Entity Recognition fällt. Vage Ortsreferenzen sind die physisch innerhalb des Textes und verorten Figuren oder Gegenstände und beschreiben so die Struktur der Welt.

Räume sind abhängig von der Perspektive: Ein "Bauch" oder eine "Schulter" sind normalerweise kein Ort, können aber durch das Platzieren von handlungsrelevanten Gegenständen oder durch das Betreten von Figuren zu welchen werden; so werden bei "der Ring lag im Bauch" oder der Trommler setzt sich "auf die Schulter des Riesen" diese zu Orten. Dienen sie nur einfachen Beschreibungen wie "trägt den Korb auf dem Rücken" oder "hält etwas vor die Brust", so sind diese keine Orte, da sie keine räumliche Funktion übernehmen. 

Es geht nicht um die Frage, ob ein Ausdruck theoretisch ein Ort sein kann, sondern ob er im Kontext als Raum verwendet wird. So wird im Satz "Wollt ihr mir das Haus anzünden?" das Haus nicht betreten, es spielt keine Handlung dort, sondern hier ist das Haus ein Objekt, das Besitz anzeigt. Wenn Türen, Fenster etc. geöffnet werden, so sind es Objekte, die bewegt werden. Zeigen die Formulierungen aber Ortswechsel durch Bewegung oder Blicke an ("aus dem Fenster", "durch die Tür", "über die Mauer"), so sind es Orte, an denen bzw. durch die Figuren sich bewegen.

Auch relationale Verortungen sind als Orte zu sehen, sofern sich Figuren dort bewegen oder handlungsrelevante Gegenstände sich dort befinden: "vor dem König", "zu seinen Füßen" oder "zu seiner Seite".

Keine Orte sind Distanzrelationen wie "auf hundert Schritt nahe".

Lässt sich ein Ort beiden Tags Vague Spatial Expression und Conceptual Spatial Expression zuschreiben, so ist die Ortsreferenz immer als Vague Spatial Expression zu sehen. So kann bspw. "auf dem Scheiterhaufen" zwar als Urteil anzusehen sein, dient aber ebenso als Raum, in dem die Verurteilte steht. Ebenso ist die Phrase "vors Gericht" in den Märchen meist ebenfalls räumlich zu verstehen. Hier ist entscheidend, die unmittelbaren Kontext (angrenzende Sätze) zu betrachten, denn oftmals befindet sich die Figur dann im Gerichtsgebäude. Dies gilt ebenfalls für "den Weg finden", eine Beschreibung, die weniger metaphorisch zu sehen ist, sondern als ein realer Weg, der verlorene Kinder durch den Wald führt.

## Conceptual Spatial Expression
- Tag: CONCEPTUAL_LOC

Conceptual Spatial Expression beschreibt den einen räumlichen Ausdruck, der nicht primär als physischer Ort verstanden wird. Das umfasst Metaphern, Vergleiche und Allegorien, aber auch symbolische und religiöse Räume.

Darunter fallen Phrasen wie: "auf dem Sterbebett", "an Land gehen", "auf den Weg machen", "auf dem Heimweg", "seines Weges gehen", "auf der ganzen Welt", "unter freiem Himmel (schlafen)", "in die weite Welt", "ins Feld (ziehen)", "bis ans Ende der Welt", "etwas im Herzen tragen" oder "vom Wege ab".

Die Beurteilung der religiösen Orte "Himmel" und "Hölle" hängt stark vom Kontext ab, da diese Konzepte im Märchen auch als Orte auftreten, die Figuren betreten (und sogar wieder verlassen) können, aber auch nur als Konzept gemeint sein können.

Besonders häufig treten Varianten von "Zuhause" und "nach Hause" auf; diese fallen ebenfalls in die Kategorie Conceptual Spatial Expressions, und nicht unter die physischen Orte.
