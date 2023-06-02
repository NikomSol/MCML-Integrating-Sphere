import numpy as np
from MCML.IS import MCML

if __name__ == '__main__':
    obj = [
            {'z_start': 0, 'z_end': 1, 'mu_a': 0,     'mu_s': 0.01,     'g': 1,      'n': 1},
            {'z_start': 1, 'z_end': 2, 'mu_a': 0.1,   'mu_s': 5,        'g': 0.9376, 'n': 1.4},
            {'z_start': 2, 'z_end': 3, 'mu_a': 0,     'mu_s': 0.01,     'g': 1,      'n': 1}
    ]

    cfg = {
            'N': 1000,  # in one thread
            'threads': 1,  # max cpu_count()-1

            'mode_generator': 'Surface',  # Surface // Volume (todo)
            'mode_spatial_distribution': 'Gauss',  # Gauss // Circle (todo)
            'mode_angular_distribution': 'Collimated',  # Collimated // Diffuse (todo) // HG (todo)

            'Surface_beam_diameter': 1,
            'Surface_beam_center': np.array([0, 0, 0]),
            'Surface_anisotropy_factor': 0.8,

            'mode_save': 'FIS',  # MIS (todo) // FIS (todo)
            'FIS_collimated_cosine': 0.99,
            'MIS_sphere_type': 'Thorlabs_IS200',
            'MIS_positions_table': np.linspace(0, 200, 10)
    }

    mcml = MCML(cfg, obj)
    mcml.run()
    print(mcml.save_data)
    print(mcml.get_output())
