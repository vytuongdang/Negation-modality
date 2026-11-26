# Dataset v2 full build on B200: H1 units (from scored_all C0 rows, qwen7b)

H1 units (one negative sample each): 4390, images 2660; asr_ok units 4338, images 2627
- branch: attribute 1518, count 899, existence 689, existence_closed 612, relation 620
- source: CLEVR 1250, COCO 1685, GQA 620, VAW 783
- split: test 410, train 779, train2017 1558, val 1464, val2017 127
- template_id: N1 577, N2 648, N3 627, N4 630, N5 581, N6 632, N7 643
- voice: female 2339, male 1999
- occludes_subject: False 4166, True 172
- template_sweep: False 3913, True 425
- branch x source: ('attribute', 'CLEVR') 735, ('attribute', 'VAW') 783, ('count', 'CLEVR') 515, ('count', 'COCO') 384, ('existence', 'COCO') 689, ('existence_closed', 'COCO') 612, ('relation', 'GQA') 620
