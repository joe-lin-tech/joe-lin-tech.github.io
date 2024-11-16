---
layout: page
title: notes
permalink: /notes/
description: collection of detailed notes and summaries from my academic courses
nav: true
nav_order: 4
horizontal: false
---

<!-- pages/notes.md -->
<div class="notes">
  <!-- Generate cards for each note -->
  <div class="row row-cols-1 row-cols-md-1">
  {% for notes in site.notes %}
    {% include notes.liquid %}
  {% endfor %}
  </div>
</div>
