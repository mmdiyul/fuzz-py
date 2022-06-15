import numpy as np
import skfuzzy as fuzzy
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route('/api/bmi', methods=['POST'])
def bmi():
        data = request.get_json()
        berat = data['berat']
        tinggi = data['tinggi']

        x_berat = np.arange(0, 175, 1)
        x_tinggi = np.arange(0, 225, 1)
        x_bmi = np.arange(0, 60, 0.1)

        berat_low = fuzzy.trimf(x_berat, [0, 45, 60])
        berat_med = fuzzy.trimf(x_berat, [45, 60, 90])
        berat_hig = fuzzy.trimf(x_berat, [60, 90, 175])

        tinggi_low = fuzzy.trimf(x_tinggi, [0, 145, 160])
        tinggi_med = fuzzy.trimf(x_tinggi, [145, 160, 180])
        tinggi_hig = fuzzy.trimf(x_tinggi, [160, 180, 225])

        bmi_under = fuzzy.trimf(x_bmi, [0, 15, 18])
        bmi_normal = fuzzy.trimf(x_bmi, [15, 18, 25])
        bmi_over = fuzzy.trimf(x_bmi, [18, 25, 33])
        bmi_obes = fuzzy.trimf(x_bmi, [25, 33, 60])

        berat_low_level = fuzzy.interp_membership(x_berat, berat_low, berat)
        berat_med_level = fuzzy.interp_membership(x_berat, berat_med, berat)
        berat_hig_level = fuzzy.interp_membership(x_berat, berat_hig, berat)

        tinggi_low_level = fuzzy.interp_membership(x_tinggi, tinggi_low, tinggi)
        tinggi_med_level = fuzzy.interp_membership(x_tinggi, tinggi_med, tinggi)
        tinggi_hig_level = fuzzy.interp_membership(x_tinggi, tinggi_hig, tinggi)

        active_rule1 = np.fmin(berat_low_level, tinggi_low_level)
        active_rule2 = np.fmin(berat_low_level, tinggi_med_level)
        active_rule3 = np.fmin(berat_low_level, tinggi_hig_level)
        active_rule4 = np.fmin(berat_med_level, tinggi_low_level)
        active_rule5 = np.fmin(berat_med_level, tinggi_med_level)
        active_rule6 = np.fmin(berat_med_level, tinggi_hig_level)
        active_rule7 = np.fmin(berat_hig_level, tinggi_low_level)
        active_rule8 = np.fmin(berat_hig_level, tinggi_med_level)
        active_rule9 = np.fmin(berat_hig_level, tinggi_hig_level)

        bmi_activation1 = np.fmin(active_rule1, bmi_normal)
        bmi_activation2 = np.fmin(active_rule2, bmi_under)
        bmi_activation3 = np.fmin(active_rule3, bmi_under)
        bmi_activation4 = np.fmin(active_rule4, bmi_over)
        bmi_activation5 = np.fmin(active_rule5, bmi_normal)
        bmi_activation6 = np.fmin(active_rule6, bmi_under)
        bmi_activation7 = np.fmin(active_rule7, bmi_obes)
        bmi_activation8 = np.fmin(active_rule8, bmi_obes)
        bmi_activation9 = np.fmin(active_rule9, bmi_normal)

        aggregated = np.fmax(bmi_activation1,
                             np.fmax(bmi_activation2,
                                     np.fmax(bmi_activation3,
                                             np.fmax(bmi_activation4,
                                                     np.fmax(bmi_activation5,
                                                             np.fmax(bmi_activation6,
                                                                     np.fmax(bmi_activation7,
                                                                             np.fmax(bmi_activation8, bmi_activation9))))))))

        result = fuzzy.defuzz(x_bmi, aggregated, 'centroid')
        resp = dict()
        resp['result'] = result
        return jsonify(resp)


@app.route('/api/bmr', methods=['POST'])
def bmr():
        data = request.get_json()
        bmi_value = data['bmi']
        umur = data['umur']
        aktifitas = 2

        x_bmi = np.arange(0, 60, 0.1)
        x_umur = np.arange(0, 100, 1)
        x_aktifitas = np.arange(0, 8, 1)
        x_kalori = np.arange(0, 3000, 1)

        bmi_under = fuzzy.trimf(x_bmi, [0, 15, 18])
        bmi_normal = fuzzy.trimf(x_bmi, [15, 18, 25])
        bmi_over = fuzzy.trimf(x_bmi, [18, 25, 33])
        bmi_obes = fuzzy.trimf(x_bmi, [25, 33, 60])

        umur_muda = fuzzy.trimf(x_umur, [0, 18, 25])
        umur_dewasa_a = fuzzy.trimf(x_umur, [19, 23, 35])
        umur_dewasa_b = fuzzy.trimf(x_umur, [25, 30, 65])
        umur_lansia = fuzzy.trimf(x_umur, [38, 60, 100])

        aktifitas_a = fuzzy.trimf(x_aktifitas, [0, 0, 2])
        aktifitas_b = fuzzy.trimf(x_aktifitas, [0, 1, 3])
        aktifitas_c = fuzzy.trimf(x_aktifitas, [1, 2, 6])
        aktifitas_d = fuzzy.trimf(x_aktifitas, [2, 5, 7])
        aktifitas_e = fuzzy.trimf(x_aktifitas, [5, 6, 8])

        kalori_a = fuzzy.trimf(x_kalori, [0, 800, 1400])
        kalori_b = fuzzy.trimf(x_kalori, [800, 1200, 2100])
        kalori_c = fuzzy.trimf(x_kalori, [1200, 2000, 2500])
        kalori_d = fuzzy.trimf(x_kalori, [2000, 2400, 3000])

        bmi_under_level = fuzzy.interp_membership(x_bmi, bmi_under, bmi_value)
        bmi_normal_level = fuzzy.interp_membership(x_bmi, bmi_normal, bmi_value)
        bmi_over_level = fuzzy.interp_membership(x_bmi, bmi_over, bmi_value)
        bmi_obes_level = fuzzy.interp_membership(x_bmi, bmi_obes, bmi_value)

        umur_muda_level = fuzzy.interp_membership(x_umur, umur_muda, umur)
        umur_dewasa_a_level = fuzzy.interp_membership(x_umur, umur_dewasa_a, umur)
        umur_dewasa_b_level = fuzzy.interp_membership(x_umur, umur_dewasa_b, umur)
        umur_lansia_level = fuzzy.interp_membership(x_umur, umur_lansia, umur)

        aktifitas_a_level = fuzzy.interp_membership(x_aktifitas, aktifitas_a, aktifitas)
        aktifitas_b_level = fuzzy.interp_membership(x_aktifitas, aktifitas_b, aktifitas)
        aktifitas_c_level = fuzzy.interp_membership(x_aktifitas, aktifitas_c, aktifitas)
        aktifitas_d_level = fuzzy.interp_membership(x_aktifitas, aktifitas_d, aktifitas)
        aktifitas_e_level = fuzzy.interp_membership(x_aktifitas, aktifitas_e, aktifitas)

        active_rule1 = np.fmin(bmi_under_level, np.fmin(umur_muda_level, aktifitas_a_level))
        active_rule2 = np.fmin(bmi_under_level, np.fmin(umur_muda_level, aktifitas_b_level))
        active_rule3 = np.fmin(bmi_under_level, np.fmin(umur_muda_level, aktifitas_c_level))
        active_rule4 = np.fmin(bmi_under_level, np.fmin(umur_muda_level, aktifitas_d_level))
        active_rule5 = np.fmin(bmi_under_level, np.fmin(umur_muda_level, aktifitas_e_level))
        active_rule6 = np.fmin(bmi_under_level, np.fmin(umur_dewasa_a_level, aktifitas_a_level))
        active_rule7 = np.fmin(bmi_under_level, np.fmin(umur_dewasa_a_level, aktifitas_b_level))
        active_rule8 = np.fmin(bmi_under_level, np.fmin(umur_dewasa_a_level, aktifitas_c_level))
        active_rule9 = np.fmin(bmi_under_level, np.fmin(umur_dewasa_a_level, aktifitas_d_level))
        active_rule10 = np.fmin(bmi_under_level, np.fmin(umur_dewasa_a_level, aktifitas_e_level))
        active_rule11 = np.fmin(bmi_under_level, np.fmin(umur_dewasa_b_level, aktifitas_a_level))
        active_rule12 = np.fmin(bmi_under_level, np.fmin(umur_dewasa_b_level, aktifitas_b_level))
        active_rule13 = np.fmin(bmi_under_level, np.fmin(umur_dewasa_b_level, aktifitas_c_level))
        active_rule14 = np.fmin(bmi_under_level, np.fmin(umur_dewasa_b_level, aktifitas_d_level))
        active_rule15 = np.fmin(bmi_under_level, np.fmin(umur_dewasa_b_level, aktifitas_e_level))
        active_rule16 = np.fmin(bmi_under_level, np.fmin(umur_lansia_level, aktifitas_a_level))
        active_rule17 = np.fmin(bmi_under_level, np.fmin(umur_lansia_level, aktifitas_b_level))
        active_rule18 = np.fmin(bmi_under_level, np.fmin(umur_lansia_level, aktifitas_c_level))
        active_rule19 = np.fmin(bmi_under_level, np.fmin(umur_lansia_level, aktifitas_d_level))
        active_rule20 = np.fmin(bmi_under_level, np.fmin(umur_lansia_level, aktifitas_e_level))
        active_rule21 = np.fmin(bmi_normal_level, np.fmin(umur_muda_level, aktifitas_a_level))
        active_rule22 = np.fmin(bmi_normal_level, np.fmin(umur_muda_level, aktifitas_b_level))
        active_rule23 = np.fmin(bmi_normal_level, np.fmin(umur_muda_level, aktifitas_c_level))
        active_rule24 = np.fmin(bmi_normal_level, np.fmin(umur_muda_level, aktifitas_d_level))
        active_rule25 = np.fmin(bmi_normal_level, np.fmin(umur_muda_level, aktifitas_e_level))
        active_rule26 = np.fmin(bmi_normal_level, np.fmin(umur_dewasa_a_level, aktifitas_a_level))
        active_rule27 = np.fmin(bmi_normal_level, np.fmin(umur_dewasa_a_level, aktifitas_b_level))
        active_rule28 = np.fmin(bmi_normal_level, np.fmin(umur_dewasa_a_level, aktifitas_c_level))
        active_rule29 = np.fmin(bmi_normal_level, np.fmin(umur_dewasa_a_level, aktifitas_d_level))
        active_rule30 = np.fmin(bmi_normal_level, np.fmin(umur_dewasa_a_level, aktifitas_e_level))
        active_rule31 = np.fmin(bmi_normal_level, np.fmin(umur_dewasa_b_level, aktifitas_a_level))
        active_rule32 = np.fmin(bmi_normal_level, np.fmin(umur_dewasa_b_level, aktifitas_b_level))
        active_rule33 = np.fmin(bmi_normal_level, np.fmin(umur_dewasa_b_level, aktifitas_c_level))
        active_rule34 = np.fmin(bmi_normal_level, np.fmin(umur_dewasa_b_level, aktifitas_d_level))
        active_rule35 = np.fmin(bmi_normal_level, np.fmin(umur_dewasa_b_level, aktifitas_e_level))
        active_rule36 = np.fmin(bmi_normal_level, np.fmin(umur_lansia_level, aktifitas_a_level))
        active_rule37 = np.fmin(bmi_normal_level, np.fmin(umur_lansia_level, aktifitas_b_level))
        active_rule38 = np.fmin(bmi_normal_level, np.fmin(umur_lansia_level, aktifitas_c_level))
        active_rule39 = np.fmin(bmi_normal_level, np.fmin(umur_lansia_level, aktifitas_d_level))
        active_rule40 = np.fmin(bmi_normal_level, np.fmin(umur_lansia_level, aktifitas_e_level))
        active_rule41 = np.fmin(bmi_over_level, np.fmin(umur_muda_level, aktifitas_a_level))
        active_rule42 = np.fmin(bmi_over_level, np.fmin(umur_muda_level, aktifitas_b_level))
        active_rule43 = np.fmin(bmi_over_level, np.fmin(umur_muda_level, aktifitas_c_level))
        active_rule44 = np.fmin(bmi_over_level, np.fmin(umur_muda_level, aktifitas_d_level))
        active_rule45 = np.fmin(bmi_over_level, np.fmin(umur_muda_level, aktifitas_e_level))
        active_rule46 = np.fmin(bmi_over_level, np.fmin(umur_dewasa_a_level, aktifitas_a_level))
        active_rule47 = np.fmin(bmi_over_level, np.fmin(umur_dewasa_a_level, aktifitas_b_level))
        active_rule48 = np.fmin(bmi_over_level, np.fmin(umur_dewasa_a_level, aktifitas_c_level))
        active_rule49 = np.fmin(bmi_over_level, np.fmin(umur_dewasa_a_level, aktifitas_d_level))
        active_rule50 = np.fmin(bmi_over_level, np.fmin(umur_dewasa_a_level, aktifitas_e_level))
        active_rule51 = np.fmin(bmi_over_level, np.fmin(umur_dewasa_b_level, aktifitas_a_level))
        active_rule52 = np.fmin(bmi_over_level, np.fmin(umur_dewasa_b_level, aktifitas_b_level))
        active_rule53 = np.fmin(bmi_over_level, np.fmin(umur_dewasa_b_level, aktifitas_c_level))
        active_rule54 = np.fmin(bmi_over_level, np.fmin(umur_dewasa_b_level, aktifitas_d_level))
        active_rule55 = np.fmin(bmi_over_level, np.fmin(umur_dewasa_b_level, aktifitas_e_level))
        active_rule56 = np.fmin(bmi_over_level, np.fmin(umur_lansia_level, aktifitas_a_level))
        active_rule57 = np.fmin(bmi_over_level, np.fmin(umur_lansia_level, aktifitas_b_level))
        active_rule58 = np.fmin(bmi_over_level, np.fmin(umur_lansia_level, aktifitas_c_level))
        active_rule59 = np.fmin(bmi_over_level, np.fmin(umur_lansia_level, aktifitas_d_level))
        active_rule60 = np.fmin(bmi_over_level, np.fmin(umur_lansia_level, aktifitas_e_level))
        active_rule61 = np.fmin(bmi_obes_level, np.fmin(umur_muda_level, aktifitas_a_level))
        active_rule62 = np.fmin(bmi_obes_level, np.fmin(umur_muda_level, aktifitas_b_level))
        active_rule63 = np.fmin(bmi_obes_level, np.fmin(umur_muda_level, aktifitas_c_level))
        active_rule64 = np.fmin(bmi_obes_level, np.fmin(umur_muda_level, aktifitas_d_level))
        active_rule65 = np.fmin(bmi_obes_level, np.fmin(umur_muda_level, aktifitas_e_level))
        active_rule66 = np.fmin(bmi_obes_level, np.fmin(umur_dewasa_a_level, aktifitas_a_level))
        active_rule67 = np.fmin(bmi_obes_level, np.fmin(umur_dewasa_a_level, aktifitas_b_level))
        active_rule68 = np.fmin(bmi_obes_level, np.fmin(umur_dewasa_a_level, aktifitas_c_level))
        active_rule69 = np.fmin(bmi_obes_level, np.fmin(umur_dewasa_a_level, aktifitas_d_level))
        active_rule70 = np.fmin(bmi_obes_level, np.fmin(umur_dewasa_a_level, aktifitas_e_level))
        active_rule71 = np.fmin(bmi_obes_level, np.fmin(umur_dewasa_b_level, aktifitas_a_level))
        active_rule72 = np.fmin(bmi_obes_level, np.fmin(umur_dewasa_b_level, aktifitas_b_level))
        active_rule73 = np.fmin(bmi_obes_level, np.fmin(umur_dewasa_b_level, aktifitas_c_level))
        active_rule74 = np.fmin(bmi_obes_level, np.fmin(umur_dewasa_b_level, aktifitas_d_level))
        active_rule75 = np.fmin(bmi_obes_level, np.fmin(umur_dewasa_b_level, aktifitas_e_level))
        active_rule76 = np.fmin(bmi_obes_level, np.fmin(umur_lansia_level, aktifitas_a_level))
        active_rule77 = np.fmin(bmi_obes_level, np.fmin(umur_lansia_level, aktifitas_b_level))
        active_rule78 = np.fmin(bmi_obes_level, np.fmin(umur_lansia_level, aktifitas_c_level))
        active_rule79 = np.fmin(bmi_obes_level, np.fmin(umur_lansia_level, aktifitas_d_level))
        active_rule80 = np.fmin(bmi_obes_level, np.fmin(umur_lansia_level, aktifitas_e_level))

        bmr_activation1 = np.fmin(active_rule1, kalori_a)
        bmr_activation2 = np.fmin(active_rule2, kalori_b)
        bmr_activation3 = np.fmin(active_rule3, kalori_c)
        bmr_activation4 = np.fmin(active_rule4, kalori_c)
        bmr_activation5 = np.fmin(active_rule5, kalori_d)
        bmr_activation6 = np.fmin(active_rule6, kalori_b)
        bmr_activation7 = np.fmin(active_rule7, kalori_c)
        bmr_activation8 = np.fmin(active_rule8, kalori_c)
        bmr_activation9 = np.fmin(active_rule9, kalori_d)
        bmr_activation10 = np.fmin(active_rule10, kalori_d)
        bmr_activation11 = np.fmin(active_rule11, kalori_a)
        bmr_activation12 = np.fmin(active_rule12, kalori_a)
        bmr_activation13 = np.fmin(active_rule13, kalori_b)
        bmr_activation14 = np.fmin(active_rule14, kalori_c)
        bmr_activation15 = np.fmin(active_rule15, kalori_d)
        bmr_activation16 = np.fmin(active_rule16, kalori_a)
        bmr_activation17 = np.fmin(active_rule17, kalori_a)
        bmr_activation18 = np.fmin(active_rule18, kalori_b)
        bmr_activation19 = np.fmin(active_rule19, kalori_c)
        bmr_activation20 = np.fmin(active_rule20, kalori_c)
        bmr_activation21 = np.fmin(active_rule21, kalori_b)
        bmr_activation22 = np.fmin(active_rule22, kalori_c)
        bmr_activation23 = np.fmin(active_rule23, kalori_c)
        bmr_activation24 = np.fmin(active_rule24, kalori_d)
        bmr_activation25 = np.fmin(active_rule25, kalori_d)
        bmr_activation26 = np.fmin(active_rule26, kalori_b)
        bmr_activation27 = np.fmin(active_rule27, kalori_c)
        bmr_activation28 = np.fmin(active_rule28, kalori_c)
        bmr_activation29 = np.fmin(active_rule29, kalori_d)
        bmr_activation30 = np.fmin(active_rule30, kalori_d)
        bmr_activation31 = np.fmin(active_rule31, kalori_a)
        bmr_activation32 = np.fmin(active_rule32, kalori_b)
        bmr_activation33 = np.fmin(active_rule33, kalori_c)
        bmr_activation34 = np.fmin(active_rule34, kalori_d)
        bmr_activation35 = np.fmin(active_rule35, kalori_d)
        bmr_activation36 = np.fmin(active_rule36, kalori_a)
        bmr_activation37 = np.fmin(active_rule37, kalori_b)
        bmr_activation38 = np.fmin(active_rule38, kalori_b)
        bmr_activation39 = np.fmin(active_rule39, kalori_c)
        bmr_activation40 = np.fmin(active_rule40, kalori_d)
        bmr_activation41 = np.fmin(active_rule41, kalori_b)
        bmr_activation42 = np.fmin(active_rule42, kalori_c)
        bmr_activation43 = np.fmin(active_rule43, kalori_c)
        bmr_activation44 = np.fmin(active_rule44, kalori_d)
        bmr_activation45 = np.fmin(active_rule45, kalori_d)
        bmr_activation46 = np.fmin(active_rule46, kalori_b)
        bmr_activation47 = np.fmin(active_rule47, kalori_b)
        bmr_activation48 = np.fmin(active_rule48, kalori_c)
        bmr_activation49 = np.fmin(active_rule49, kalori_c)
        bmr_activation50 = np.fmin(active_rule50, kalori_d)
        bmr_activation51 = np.fmin(active_rule51, kalori_a)
        bmr_activation52 = np.fmin(active_rule52, kalori_b)
        bmr_activation53 = np.fmin(active_rule53, kalori_c)
        bmr_activation54 = np.fmin(active_rule54, kalori_c)
        bmr_activation55 = np.fmin(active_rule55, kalori_d)
        bmr_activation56 = np.fmin(active_rule56, kalori_a)
        bmr_activation57 = np.fmin(active_rule57, kalori_b)
        bmr_activation58 = np.fmin(active_rule58, kalori_c)
        bmr_activation59 = np.fmin(active_rule59, kalori_c)
        bmr_activation60 = np.fmin(active_rule60, kalori_d)
        bmr_activation61 = np.fmin(active_rule61, kalori_b)
        bmr_activation62 = np.fmin(active_rule62, kalori_c)
        bmr_activation63 = np.fmin(active_rule63, kalori_d)
        bmr_activation64 = np.fmin(active_rule64, kalori_d)
        bmr_activation65 = np.fmin(active_rule65, kalori_d)
        bmr_activation66 = np.fmin(active_rule66, kalori_b)
        bmr_activation67 = np.fmin(active_rule67, kalori_c)
        bmr_activation68 = np.fmin(active_rule68, kalori_d)
        bmr_activation69 = np.fmin(active_rule69, kalori_d)
        bmr_activation70 = np.fmin(active_rule70, kalori_d)
        bmr_activation71 = np.fmin(active_rule71, kalori_b)
        bmr_activation72 = np.fmin(active_rule72, kalori_c)
        bmr_activation73 = np.fmin(active_rule73, kalori_c)
        bmr_activation74 = np.fmin(active_rule74, kalori_d)
        bmr_activation75 = np.fmin(active_rule75, kalori_d)
        bmr_activation76 = np.fmin(active_rule76, kalori_a)
        bmr_activation77 = np.fmin(active_rule77, kalori_b)
        bmr_activation78 = np.fmin(active_rule78, kalori_c)
        bmr_activation79 = np.fmin(active_rule79, kalori_d)
        bmr_activation80 = np.fmin(active_rule80, kalori_d)

        aggregated = np.fmax(bmr_activation1, np.fmax(bmr_activation2, np.fmax(bmr_activation3, np.fmax(bmr_activation4,
                                                                                                        np.fmax(bmr_activation5,
                                                                                                                np.fmax(bmr_activation6,
                                                                                                                        np.fmax(bmr_activation7,
                                                                                                                                np.fmax(bmr_activation8,
                                                                                                                                        np.fmax(bmr_activation9,
                                                                                                                                                np.fmax(bmr_activation10,
                                                                                                                                                        np.fmax(bmr_activation11,
                                                                                                                                                                np.fmax(bmr_activation12,
                                                                                                                                                                        np.fmax(bmr_activation13,
                                                                                                                                                                                np.fmax(bmr_activation14,
                                                                                                                                                                                        np.fmax(bmr_activation15,
                                                                                                                                                                                                np.fmax(bmr_activation16,
                                                                                                                                                                                                        np.fmax(bmr_activation17,
                                                                                                                                                                                                                np.fmax(bmr_activation18,
                                                                                                                                                                                                                        np.fmax(bmr_activation19,
                                                                                                                                                                                                                                np.fmax(bmr_activation20,
                                                                                                                                                                                                                                        np.fmax(bmr_activation21,
                                                                                                                                                                                                                                                np.fmax(bmr_activation22,
                                                                                                                                                                                                                                                        np.fmax(bmr_activation23,
                                                                                                                                                                                                                                                                np.fmax(bmr_activation24,
                                                                                                                                                                                                                                                                        np.fmax(bmr_activation25,
                                                                                                                                                                                                                                                                                np.fmax(bmr_activation26,
                                                                                                                                                                                                                                                                                        np.fmax(bmr_activation27,
                                                                                                                                                                                                                                                                                                np.fmax(bmr_activation28,
                                                                                                                                                                                                                                                                                                        np.fmax(bmr_activation29,
                                                                                                                                                                                                                                                                                                                np.fmax(bmr_activation30,
                                                                                                                                                                                                                                                                                                                        np.fmax(bmr_activation31,
                                                                                                                                                                                                                                                                                                                                np.fmax(bmr_activation32,
                                                                                                                                                                                                                                                                                                                                        np.fmax(bmr_activation33,
                                                                                                                                                                                                                                                                                                                                                np.fmax(bmr_activation34,
                                                                                                                                                                                                                                                                                                                                                        np.fmax(bmr_activation35,
                                                                                                                                                                                                                                                                                                                                                                np.fmax(bmr_activation36,
                                                                                                                                                                                                                                                                                                                                                                        np.fmax(bmr_activation37,
                                                                                                                                                                                                                                                                                                                                                                                np.fmax(bmr_activation38,
                                                                                                                                                                                                                                                                                                                                                                                        np.fmax(bmr_activation39,
                                                                                                                                                                                                                                                                                                                                                                                                np.fmax(bmr_activation40,
                                                                                                                                                                                                                                                                                                                                                                                                        np.fmax(bmr_activation41,
                                                                                                                                                                                                                                                                                                                                                                                                                np.fmax(bmr_activation42,
                                                                                                                                                                                                                                                                                                                                                                                                                        np.fmax(bmr_activation43,
                                                                                                                                                                                                                                                                                                                                                                                                                                np.fmax(bmr_activation44,
                                                                                                                                                                                                                                                                                                                                                                                                                                        np.fmax(bmr_activation45,
                                                                                                                                                                                                                                                                                                                                                                                                                                                np.fmax(bmr_activation46,
                                                                                                                                                                                                                                                                                                                                                                                                                                                        np.fmax(bmr_activation47,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                np.fmax(bmr_activation48,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                        np.fmax(bmr_activation49,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                np.fmax(bmr_activation50,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        np.fmax(bmr_activation51,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                np.fmax(bmr_activation52,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        np.fmax(bmr_activation53,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                np.fmax(bmr_activation54,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        np.fmax(bmr_activation55,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                np.fmax(bmr_activation56,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        np.fmax(bmr_activation57,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                np.fmax(bmr_activation58,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        np.fmax(bmr_activation59,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                np.fmax(bmr_activation60,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        np.fmax(bmr_activation61,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                np.fmax(bmr_activation62,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        np.fmax(bmr_activation63,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                np.fmax(bmr_activation64,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        np.fmax(bmr_activation65,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                np.fmax(bmr_activation66,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        np.fmax(bmr_activation67,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                np.fmax(bmr_activation68,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        np.fmax(bmr_activation69,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                np.fmax(bmr_activation70,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        np.fmax(bmr_activation71,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                np.fmax(bmr_activation72,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        np.fmax(bmr_activation73,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                np.fmax(bmr_activation74,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        np.fmax(bmr_activation75,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                np.fmax(bmr_activation76,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        np.fmax(bmr_activation77,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                np.fmax(bmr_activation78,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        np.fmax(bmr_activation79,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                bmr_activation80)))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))
        result = fuzzy.defuzz(x_kalori, aggregated, 'centroid')
        resp = dict()
        resp['result'] = result
        return jsonify(resp)


if __name__ == '__main__':
    app.run()
