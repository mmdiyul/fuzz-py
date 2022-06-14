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


if __name__ == '__main__':
    app.run()
