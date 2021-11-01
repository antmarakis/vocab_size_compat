####### SETUP
WORKDIR="/mounts/work/antmarakis/emnlp2021"
BIBLEDIR="/nfs/datc/pbc/"

mkdir -p ${WORKDIR}
mkdir -p ${WORKDIR}/data
mkdir -p ${WORKDIR}/embeddings
mkdir -p ${WORKDIR}/results
mkdir -p ${WORKDIR}/data_new
mkdir -p ${WORKDIR}/plots

####### CREATING BIBLE ISOMORPHY PLOTS
# list of all newworld editions
editions="afr_newworld ewe_newworld ibo_newworld nob_newworld sag_newworld tha_newworld aln_newworld fij_newworld ilo_newworld npi_newworld sin_newworld tir_newworld amh_newworld fin_newworld ind_newworld nso_newworld slk_newworld tpi_newworld arb_newworld fra_newworld ita_newworld nya_newworld slv_newworld tso_newworld aze_newworld gaa_newworld jpn_newworld oss_newworld smo_newworld tur_newworld bem_newworld gug_newworld kan_newworld pag_newworld sna_newworld twi_newworld bul_newworld guw_newworld kaz_newworld1984 pan_newworld sot_newworld tzo_newworld ceb_newworld hat_newworld2007 khm_newworld pap_newworld spa_newworld uzb_newworld ces_newworld hat_newworld2015 kik_newworld pes_newworld srn_newworld ven_newworld cmn_newworld heb_newworld kin_newworld pis_newworld srp_newworld vie_newworld dan_newworld hil_newworld2007 kir_newworld plt_newworld ssw_newworld war_newworld deu_newworld hil_newworld2014 kor_newworld1999 pol_newworld swe_newworld xho_newworld efi_newworld hin_newworld kor_newworld2014 por_newworld1996 swh_newworld yor_newworld ell_newworld hmo_newworld lin_newworld por_newworld2015 yua_newworld eng_newworld1984 hrv_newworld mlt_newworld ron_newworld tam_newworld zho_newworld eng_newworld2013 hun_newworld mya_newworld run_newworld tdt_newworld zul_newworld est_newworld2009 hye_newworld nld_newworld rus_newworld tgl_newworld"
editionscomma="afr_newworld,ewe_newworld,ibo_newworld,nob_newworld,sag_newworld,tha_newworld,aln_newworld,fij_newworld,ilo_newworld,npi_newworld,sin_newworld,tir_newworld,amh_newworld,fin_newworld,ind_newworld,nso_newworld,slk_newworld,tpi_newworld,arb_newworld,fra_newworld,ita_newworld,nya_newworld,slv_newworld,tso_newworld,aze_newworld,gaa_newworld,jpn_newworld,oss_newworld,smo_newworld,tur_newworld,bem_newworld,gug_newworld,kan_newworld,pag_newworld,sna_newworld,twi_newworld,bul_newworld,guw_newworld,kaz_newworld1984,pan_newworld,sot_newworld,tzo_newworld,ceb_newworld,hat_newworld2007,khm_newworld,pap_newworld,spa_newworld,uzb_newworld,ces_newworld,hat_newworld2015,kik_newworld,pes_newworld,srn_newworld,ven_newworld,cmn_newworld,heb_newworld,kin_newworld,pis_newworld,srp_newworld,vie_newworld,dan_newworld,hil_newworld2007,kir_newworld,plt_newworld,ssw_newworld,war_newworld,deu_newworld,hil_newworld2014,kor_newworld1999,pol_newworld,swe_newworld,xho_newworld,efi_newworld,hin_newworld,kor_newworld2014,por_newworld1996,swh_newworld,yor_newworld,ell_newworld,hmo_newworld,lin_newworld,por_newworld2015,yua_newworld,eng_newworld1984,hrv_newworld,mlt_newworld,ron_newworld,tam_newworld,zho_newworld,eng_newworld2013,hun_newworld,mya_newworld,run_newworld,tdt_newworld,zul_newworld,est_newworld2009,hye_newworld,nld_newworld,rus_newworld,tgl_newworld"

# large editions contain old and new editions
largeeditions='afr_newworld ibo_newworld nob_newworld aln_newworld fij_newworld ilo_newworld sin_newworld fin_newworld ind_newworld nso_newworld slk_newworld arb_newworld fra_newworld ita_newworld nya_newworld slv_newworld tso_newworld aze_newworld jpn_newworld oss_newworld smo_newworld tur_newworld bem_newworld sna_newworld twi_newworld bul_newworld sot_newworld ceb_newworld spa_newworld ces_newworld srn_newworld cmn_newworld kin_newworld srp_newworld dan_newworld kir_newworld deu_newworld kor_newworld1999 kor_newworld2014 pol_newworld swe_newworld xho_newworld efi_newworld por_newworld1996 swh_newworld yor_newworld ell_newworld lin_newworld eng_newworld1984 eng_newworld2013 hrv_newworld mlt_newworld ron_newworld zho_newworld hun_newworld run_newworld zul_newworld hye_newworld nld_newworld rus_newworld tgl_newworld'

# set experiment ID
exid=00

# get data
python -m utils.prepare_data \
--output_dir ${WORKDIR}/data_new \
--editions ${editionscomma}

#pip install --upgrade tokenizers
# get vocab sizes
python -m get_vocab_sizes \
--pbc ${BIBLEDIR} \
--editions ${editionscomma} \
--outpath ${WORKDIR}/sizes,${exid}.txt

# create embeddings
mkdir -p ${WORKDIR}/embeddings,${exid}
#pip install --upgrade tokenizers==0.8.0
for edition in ${editions}
do
	python -m main \
	--corpus ${WORKDIR}/data_new \
	--output_dir ${WORKDIR}/embeddings,${exid} \
	--result_file ${WORKDIR}/results/${exid}.txt \
	--l1 eng_newworld2013 \
	--l2 ${edition} \
	--exid ${exid} \
	--embed_onlyl2 \
	--sizes ${WORKDIR}/sizes,${exid}.txt \
	--do_vocabembeds
done

# compute measures
for edition in ${editions}
do
	python -m main \
	--corpus ${WORKDIR}/data_new \
	--output_dir ${WORKDIR}/embeddings,${exid} \
	--result_file ${WORKDIR}/results/${exid}.txt \
	--l1 eng_newworld2013 \
	--l2 ${edition} \
	--exid ${exid} \
	--sizes ${WORKDIR}/sizes,${exid}.txt \
	--do_isoeval 
done

# create plots
editions='afr_newworld ibo_newworld nob_newworld aln_newworld fij_newworld ilo_newworld sin_newworld fin_newworld ind_newworld nso_newworld slk_newworld arb_newworld fra_newworld ita_newworld nya_newworld slv_newworld tso_newworld aze_newworld jpn_newworld oss_newworld smo_newworld tur_newworld bem_newworld sna_newworld twi_newworld bul_newworld sot_newworld ceb_newworld spa_newworld ces_newworld srn_newworld cmn_newworld kin_newworld srp_newworld dan_newworld kir_newworld deu_newworld kor_newworld1999 kor_newworld2014 pol_newworld swe_newworld xho_newworld efi_newworld por_newworld1996 swh_newworld yor_newworld ell_newworld lin_newworld eng_newworld1984 eng_newworld2013 hrv_newworld mlt_newworld ron_newworld zho_newworld hun_newworld run_newworld zul_newworld hye_newworld nld_newworld rus_newworld tgl_newworld'

for edition in ${editions}
do
	#for task in svg cond-hm econd-hm
	for task in svg
	do
		mkdir -p ${WORKDIR}/results/plots3,${exid}-stars,${task}
		echo $edition
		python -m figure \
		--infile ${WORKDIR}/results/${exid}.txt \
		--outfile ${WORKDIR}/results/plots3,${exid}-stars,${task}/${exid},eng_newworld2013,${edition}.pdf \
		--exid ${exid} \
		--l1 eng_newworld2013 \
		--l2 ${edition} \
		--task ${task}
	done
done



####### WIKIPEDIA PLOTS


# /mounts/Users/cisintern/antmarakis/data/wikipedia/wikipedia_${lang}_1G.train

exid=wiki5
langs="en ru zh el"


mkdir ${WORKDIR}/data,wiki
# get clean data
for lang in ${langs}
do
python -m utils.prepare_data \
--input /mounts/Users/cisintern/antmarakis/data/wikipedia/wikipedia_${lang}_1G.train \
--output ${WORKDIR}/data,wiki/${lang}.txt \
--take_max_n 1000000 \
--samplingrate 0.2 \
--sentence_tokenize
done

#pip install --upgrade tokenizers
for lang in ${langs}
do
python -m get_vocab_sizes \
--data /mounts/Users/cisintern/antmarakis/data/wikipedia/wikipedia_${lang}_1G.train \
--outpath ${WORKDIR}/sizes,${exid}.txt \
--editions ${lang} \
--max_vocab 100000
done

mkdir -p ${WORKDIR}/embeddings,${exid}
for lang in ${langs}
do
	python -m main \
	--corpus ${WORKDIR}/data,wiki/ \
	--output_dir ${WORKDIR}/embeddings,${exid} \
	--result_file ${WORKDIR}/results/${exid}.txt \
	--l1 en \
	--l2 ${lang} \
	--exid ${exid} \
	--embed_onlyl2 \
	--sizes ${WORKDIR}/sizes,${exid}.txt \
	--do_vocabembeds
done

for lang in ${langs}
do
	python -m main \
	--corpus ${WORKDIR}/data,wiki/${lang}.txt \
	--output_dir ${WORKDIR}/embeddings,${exid} \
	--result_file ${WORKDIR}/results/${exid}.txt \
	--l1 en \
	--l2 ${lang} \
	--exid ${exid} \
	--sizes ${WORKDIR}/sizes,${exid}.txt \
	--do_isoeval 
done

#mkdir -p ${WORKDIR}/results/plots,${exid}
for lang in ${langs}
do
	#for task in svg cond-hm econd-hm
	for task in svg
	do
		mkdir -p ${WORKDIR}/results/plots,${exid}-stars,${task}
		echo $lang
		python -m figure \
		--infile ${WORKDIR}/results/${exid}.txt \
		--outfile ${WORKDIR}/results/plots,${exid}-stars,${task}/${exid},en,${lang}.pdf \
		--exid ${exid} \
		--l1 en \
		--l2 ${lang} \
		--task ${task}
	done
done



####### mBERT COMPRESSION RATES ON REAL DATA (for Correlation analysis)

wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-vocab.txt
python -m mbert \
--langs ar

python -m mbert --langs en,el,ru,zh --firstn 100

# correlation analysis
# https://docs.google.com/spreadsheets/d/181M3QWs23ZCzshSPZeRgRKrdt_mr_G8lB3b5QaNrnJs/edit?usp=sharing



####### PLOT THE COMPRESSION RATES FOR DIFFERENT LANGUAGES

python -m plot_measures \
--editions eng_newworld2013 \
--sizes 84,100,129,147,172,207,264,369,638,3941,5000,10000,18865 \
--outfile ${WORKDIR}/plots

python -m plot_measures \
--editions zho_newworld \
--sizes 3413,3450,3500,3750,4000,5000,7500,10000,20000,33784 \
--outfile ${WORKDIR}/plots

python -m plot_measures \
--editions eng_newworld2013 \
--sizes 147,172,207,264,369,638,3941,7500,10000,15000,18865 \
--outfile ${WORKDIR}/plots

python -m plot_measures \
--editions zho_newworld \
--sizes 5000,5100,5250,5500,6000,7500,10000,20000,30000,40000,50000,75000 \
--outfile ${WORKDIR}/plots

python -m plot_measures \
--editions rus_newworld \
--sizes 160,192,239,314,452,778,2384,5000,10000,25000,40000,59342 \
--outfile ${WORKDIR}/plots

python -m plot_measures \
--editions ell_newworld \
--sizes 169,206,261,350,518,930,3099,5000,10000,25000,51079 \
--outfile ${WORKDIR}/plots
