from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Pt


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "entrega_professor" / "artigo_edutech_vision_grupo3.docx"


def add_paragraph(document: Document, text: str) -> None:
    paragraph = document.add_paragraph(text)
    paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    paragraph.paragraph_format.space_after = Pt(6)


def main() -> None:
    document = Document()
    section = document.sections[0]
    section.top_margin = Cm(2)
    section.bottom_margin = Cm(2)
    section.left_margin = Cm(2)
    section.right_margin = Cm(2)

    normal = document.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(11)

    title = document.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run(
        "EduTech Vision: Monitoramento de Atenção, Postura e Engajamento "
        "por Processamento Digital de Imagens"
    )
    run.bold = True
    run.font.size = Pt(16)
    authors = document.add_paragraph("Fernando Cobianchi e Samuel Bernini")
    authors.alignment = WD_ALIGN_PARAGRAPH.CENTER
    authors.runs[0].bold = True
    institution = document.add_paragraph(
        "Bacharelado em Ciência da Computação - UNEMAT, Câmpus Universitário de Rondonópolis"
    )
    institution.alignment = WD_ALIGN_PARAGRAPH.CENTER

    document.add_heading("Resumo", level=1)
    add_paragraph(
        document,
        "Este artigo apresenta o EduTech Vision, um sistema de visão computacional "
        "para contextos educacionais. O produto opera em modo individual por webcam "
        "e em modo plateia agregado, combinando detecção de faces, landmarks, EAR/MAR, "
        "estimativa de pose de cabeça e suavização temporal. A principal melhoria usa "
        "OpenCV YuNet antes de MediaPipe. Em benchmark smoke com WIDER FACE e vídeos "
        "públicos, o recall de face subiu de 0,193 para 0,592 no modo individual e de "
        "0,011 para 0,443 no modo plateia, mantendo 29,0 e 74,5 FPS em 960x540.",
    )

    sections = [
        (
            "1. Introdução",
            "O EduTech Vision foi desenvolvido para demonstrar PDI aplicado a atenção, "
            "postura e engajamento. No modo individual, evidencia fadiga ocular, bocejo, "
            "inclinação e desatenção sustentada. No modo plateia, estima a proporção de "
            "faces com pose orientada ao palco em janelas de tempo. Os testes iniciais "
            "revelaram perda de rostos distantes ao aplicar MediaPipe diretamente ao quadro; "
            "a implementação passou a usar prior art para localizar as regiões antes dos landmarks.",
        ),
        (
            "2. Pipeline De PDI",
            "O pipeline padrão executa captura de vídeo, detecção YuNet, NMS, recortes "
            "ampliados, landmarks MediaPipe, EAR, MAR, pose com solvePnP, média móvel, "
            "estados sustentados, painel e logs. O baseline mediapipe preserva a execução "
            "direta no quadro; o perfil research troca apenas o detector inicial por SCRFD. "
            "No modo plateia, faces detectadas sem landmarks suficientes permanecem visíveis, "
            "mas não entram no cálculo de orientação ou engajamento.",
        ),
        (
            "3. Avaliação Reprodutível",
            "A avaliação usa WIDER FACE validation como conjunto anotado e aplica variações "
            "controladas de iluminação, blur, distância e oclusão. Vídeos públicos de sala "
            "de aula em 960x540 medem desempenho do caminho completo. O perfil padrao e "
            "aceito somente com ganho de pelo menos 0,20 em faces pequenas ou distantes, "
            "precisão mínima de 0,85 e FPS médio mínimo de 15 no individual e 10 na plateia.",
        ),
    ]
    for heading, text in sections:
        document.add_heading(heading, level=1)
        add_paragraph(document, text)

    table = document.add_table(rows=1, cols=7)
    table.style = "Table Grid"
    headers = ["Modo", "Perfil", "Recall face", "Recall landmarks", "Precisão", "F1", "FPS"]
    for cell, value in zip(table.rows[0].cells, headers):
        cell.text = value
    rows = [
        ["Individual", "MediaPipe", "0,193", "0,193", "1,000", "0,324", "242,4"],
        ["Individual", "YuNet", "0,592", "0,294", "0,894", "0,712", "29,0"],
        ["Plateia", "MediaPipe", "0,011", "0,011", "1,000", "0,022", "491,6"],
        ["Plateia", "YuNet", "0,443", "0,089", "0,851", "0,583", "74,5"],
    ]
    for row in rows:
        cells = table.add_row().cells
        for cell, value in zip(cells, row):
            cell.text = value

    add_paragraph(
        document,
        "YuNet passou nos gates dos dois modos: ganhou 0,399 de recall no Individual "
        "e 0,432 na Plateia contra o baseline. Na auditoria HQ em vídeos completos, "
        "a configuração enhanced_c60_m24 teve erro médio de 0,636 rosto; portanto "
        "é o detector padrão da demonstração.",
    )
    for heading, text in [
        (
            "4. Demonstração E Limitações",
            "A interface seleciona modo e detector, mostra caixa facial, landmarks, barras, "
            "estados e linha do tempo, e abre relatórios. O setup em Windows 10/11, Linux e macOS ocorre "
            "por um único comando, run.bat ou run.sh. Iluminação extrema, oclusão, ângulo lateral e "
            "rostos muito pequenos ainda limitam landmarks e pose; as métricas devem ser "
            "interpretadas como indicadores visuais de uma demonstração técnica.",
        ),
        (
            "5. Conclusão",
            "A introdução de YuNet antes de MediaPipe corrigiu o principal gargalo observado, "
            "especialmente na cena de plateia, sem inviabilizar tempo real. A comparacao com "
            "baseline e a auditoria HQ torna a decisão técnica verificável e defensável na apresentação.",
        ),
    ]:
        document.add_heading(heading, level=1)
        add_paragraph(document, text)

    document.add_heading("Referências", level=1)
    for reference in [
        "Google AI Edge. MediaPipe Face Landmarker. https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python. Acesso em: maio de 2026.",
        "OpenCV Zoo. YuNet Face Detection. https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet. Acesso em: maio de 2026.",
        "Yang, S.; Luo, P.; Loy, C. C.; Tang, X. WIDER FACE: A Face Detection Benchmark. CVPR, 2016.",
        "Szeliski, R. Computer Vision: Algorithms and Applications. 2. ed. Springer, 2022.",
    ]:
        add_paragraph(document, reference)

    document.save(OUTPUT)
    print(f"DOCX gerado: {OUTPUT}")


if __name__ == "__main__":
    main()
