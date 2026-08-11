# Harbinger

<div align = "center">

[![Join our community (Element Matrix)!](https://img.shields.io/badge/chat-on%20matrix-51bb9c?style=for-the-badge)](https://matrix.to/#/#confoederatio:matrix.confoederatio.org) [![Join our community (Discord)!](https://img.shields.io/discord/548994743925997570?label=Discord&style=for-the-badge)](https://discord.gg/89kQY2KFQz) ![](https://img.shields.io/github/languages/code-size/Confoederatio/Naissance?style=for-the-badge)

</div>
<div align = "center">
<img src = "https://i.postimg.cc/3ND2B1zL/crd-coat-of-arms-logo.png" height = "52"> <img src = "https://i.postimg.cc/0NCrhpK4/naissance-logo.png" height = "52">
</div><br>

### Abstract.

**Harbinger** is a series of CV tools/pipelines currently developed by Special Research Groups (SRGs) within CRD, capable of scraping adversarial, noisy sources, and restructuring them into clean, georeferenced vector geometries with text. Unlike other toolkits, Harbinger does not care if sources come with metadata, are highly compressed, or are very noisy.

These features are intended for final use in [Naissance HGIS](https://github.com/ConfoederatioVF/Naissance). As an end-to-end pipeline, Harbinger is intended for creating geographic datasets from raster/vector sources, which future spatial AI models inside Confoederatio can train on.

---

<img src = "https://i.postimg.cc/c4LBZSNh/63-harbinger.jpg" width = "100%">
<div align = "center"><i>A spectre is haunting mapping ... the spectre of big vector ...</i></div>

---

Current tools included as part of **Harbinger**/Naissance HGIS:

1. [Harbinger.Deprojector](https://github.com/ConfoederatioVF/Harbinger.Deprojector): Converts any arbitrary source image into an arbitrary projection, without knowing what projection the source input is in.
2. [Harbinger.Segmentation](https://github.com/ConfoederatioVF/Harbinger.Segmentation): Segmentation model for symbolic/noisy maps, satellite imagery. Also capable of rapid segmentation for real-world images and footage.
3. Harbinger.Geowarp: Part of [Naissance HGIS](https://github.com/ConfoederatioVF/Naissance). Provides 4D warping and time sync capabilities for video and image sources, as well as real-time georeferencing and manual adjustment.
