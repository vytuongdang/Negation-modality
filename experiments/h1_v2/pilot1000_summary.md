# pilot1000 (1000 questions, seed 42)

Fixed list in `pilot1000.json` (sample_ids). One negative sample per unit, asr_ok, distinct images over the whole list, excluded units dropped.

**branch**: attribute=200, count=200, existence=200, existence_closed=200, relation=200

**source**: CLEVR=199, COCO=500, GQA=200, VAW=101

**voice**: female=500, male=500

**split**: train=673, val=327

**template_id**: N1=145, N2=143, N3=143, N4=143, N5=141, N6=142, N7=143

**occludes_subject**: False=950, True=50

**branch x source**: attribute/CLEVR=99, attribute/VAW=101, count/CLEVR=100, count/COCO=100, existence/COCO=200, existence_closed/COCO=200, relation/GQA=200

**branch x voice**: attribute/female=100, attribute/male=100, count/female=100, count/male=100, existence/female=100, existence/male=100, existence_closed/female=100, existence_closed/male=100, relation/female=100, relation/male=100

**branch x split**: attribute/train=99, attribute/val=101, count/train=100, count/val=100, existence/train=157, existence/val=43, existence_closed/train=193, existence_closed/val=7, relation/train=124, relation/val=76

**H4 forms present per branch**: attribute/affirmative+exceptive=200, count/affirmative+exceptive=200, existence/exceptive=200, existence_closed/affirmative+exceptive=200, relation/affirmative+exceptive=200
