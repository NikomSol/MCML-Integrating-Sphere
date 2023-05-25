from unittest import TestCase, main
from mcml import MCML
import numpy as np

class TestMCML(TestCase):
    obj = [
        {'z_start': 0, 'z_end': 1, 'mu_a': 0,     'mu_s': 0.01,     'g': 1,      'n': 1},
        {'z_start': 1, 'z_end': 2, 'mu_a': 1.568, 'mu_s': 12.72, 'g': 0.9376, 'n': 1.4},
        {'z_start': 2, 'z_end': 3, 'mu_a': 0,     'mu_s': 0.01,     'g': 1,      'n': 1}
    ]

    cnf = {
        'N': 100000,

        'mode_generator': 'Surface',
        'Surface_spatial_distribution': 'Gauss',
        'Surface_beam_diameter': 1,
        'Surface_beam_center': np.array([0,0,0]),
        'Surface_angular_distribution': 'Collimated',

        'mode_save': 'FIS',
        'FIS_collimated_cosinus': 0.99,
    }

    mcml = MCML(cnf, obj)

    def test_modes_generator(self):
        obj = self.obj
        cnf = self.cnf

        MCML(cnf, obj)

        cnf['Surface_spatial_distribution'] = "Cyrcle"
        with self.assertRaises(ValueError):
            MCML(cnf, obj)
        cnf['Surface_spatial_distribution'] = "Gauss"

        cnf['Surface_angular_distribution'] = "Diffuse"
        with self.assertRaises(ValueError):
            MCML(cnf, obj)
        cnf['Surface_angular_distribution'] = "Collimated"

        cnf['Surface_angular_distribution'] = "HG"
        with self.assertRaises(ValueError):
            MCML(cnf, obj)
        cnf['Surface_angular_distribution'] = "Collimated"

        cnf['mode_generator'] = "Volume"
        with self.assertRaises(ValueError):
            MCML(cnf, obj)
        cnf['mode_generator'] = "Surface"

    def test_modes_save(self):
        obj = self.obj
        cnf = self.cnf
        
        MCML(cnf, obj)

        cnf['mode_save'] = "MIS"
        with self.assertRaises(ValueError):
            MCML(cnf, obj)
        cnf['mode_save'] = "FIS"

    def test_generator(self):
        mcml = self.mcml

        self.assertTrue(np.array_equal(mcml.generator()[2:],np.array([0,0,0,1,1,-1])))

    def test_turn(self):
        mcml = self.mcml
        p0 = np.array([0,0,0.5,0,0,1,1,0])
        p1 = mcml.get_func_turn()(p0)
        
        self.assertTrue(np.array_equal(p0,p1))

        p0 = np.array([0,0,1.5,0,0,1,1,1])
        p1 = mcml.get_func_turn()(p0)

        self.assertFalse(np.array_equal(p0,p1))
        self.assertTrue(np.array_equal(p0[:3],p1[:3]))
        self.assertTrue(np.array_equal(p0[-2:],p1[-2:]))
        self.assertEqual(1., p0[3]**2 + p0[4]**2 + p0[5]**2)

    def test_move(self):
        mcml = self.mcml
        p0 = np.array([0,0,0.5,0,0,1,10**-5,1])
        p1 = mcml.get_func_term()(p0)
        p2 = np.array([0,0,0.5,0,0,1,0,1])

        self.assertTrue(np.array_equal(p0,p1) or np.array_equal(p2,p1))
        

    def test_1(self):
        self.assertEqual(1,1)

    def test_2(self):
        with self.assertRaises(ValueError) as buf:
            raise(ValueError)

if __name__ =='__main__':
    main()