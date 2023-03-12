import os
import warnings

import numpy as np

import gdown
from simim.siminterface._rawsiminterface import sim_catalogs, snapshot
from simim.siminterface._sims import _checksim

a_smdpl = ['0.051120', '0.055610', '0.060130', '0.064640', '0.069120',
           '0.073620', '0.078120', '0.082620', '0.089370', '0.096750',
           '0.104000', '0.109000', '0.113000', '0.118000', '0.123000',
           '0.128000', '0.133000', '0.139000', '0.145000', '0.151000',
           '0.157000', '0.164000', '0.171000', '0.178000', '0.186000',
           '0.194000', '0.202000', '0.210000', '0.210000', '0.228000',
           '0.238000', '0.248000', '0.258000', '0.269000', '0.281000',
           '0.292000', '0.305000', '0.318000', '0.331000', '0.345000',
           '0.359000', '0.375000', '0.390000', '0.407000', '0.424000',
           '0.442000', '0.460000', '0.480000', '0.500000', '0.530000',
           '0.544400', '0.550400', '0.556300', '0.562300', '0.568400',
           '0.574300', '0.580300', '0.586400', '0.592400', '0.598300',
           '0.604300', '0.610400', '0.616300', '0.622300', '0.628400',
           '0.634400', '0.640300', '0.646400', '0.652400', '0.658300',
           '0.664300', '0.670400', '0.676300', '0.682300', '0.688400',
           '0.694400', '0.700300', '0.706400', '0.712400', '0.718300',
           '0.724300', '0.736300', '0.742300', '0.754400', '0.760300',
           '0.772400', '0.778300', '0.787300', '0.790400', '0.802300',
           '0.808400', '0.814400', '0.817300', '0.823400', '0.826300',
           '0.832400', '0.835300', '0.841400', '0.844300', '0.850400',
           '0.853300', '0.859400', '0.862300', '0.868400', '0.871300',
           '0.877300', '0.880300', '0.886300', '0.889300', '0.895300',
           '0.898400', '0.904300', '0.907400', '0.925150', '0.956000',
           '0.970710', '1.000000']
n_snap_smdpl = len(a_smdpl)

# To get this, go to the correct google drive folder and open developer tools console
# run $$("[data-id]").map((el) => 'https://drive.google.com/uc?id=' + el.getAttribute('data-id')).join(" ")
# If the drive is sorted by name, the links should be in the same order and can
# pick out the ones that are snapshots
links_smdpl = ["https://drive.google.com/uc?id=1lM_zJ5rwB0k6Qo0ku733sYlCOyTw78Ln",
    "https://drive.google.com/uc?id=1BuqMTnYSU0224HjAP0833E4wImAwizAp",
    "https://drive.google.com/uc?id=1iuAfFRgFXmQgvI-oO10AkuYYQpk-pPWF",
    "https://drive.google.com/uc?id=1QAKmVE3lsio6TywJgQB11USb2M4mWZJ-",
    "https://drive.google.com/uc?id=1CRe6AtyU47rhhuG7Y-jni4uckZP56aUj",
    "https://drive.google.com/uc?id=1-wOA0tsyMFHwN_EhGzWSL5O6JMOaPJJA",
    "https://drive.google.com/uc?id=1_I5ySfeRXOHFnue-mzAG7DkL8c1XqHeT",
    "https://drive.google.com/uc?id=139UXbXlkwBeAGz7a9nx7b2SGrcl7HmSz",
    "https://drive.google.com/uc?id=1O2gA-tcS4gc9AOWrN8J9yl2J1qLgW_yh",
    "https://drive.google.com/uc?id=1s3YN6IwUMzV2jmZxOWFWVX97Vy-av7N3",
    "https://drive.google.com/uc?id=1D--ITyxl0oHWzZcwFzt0Nj0FfqLqwao5",
    "https://drive.google.com/uc?id=1byu2Q-5VWi2JzPXppom2X8_vWqSl-OGP",
    "https://drive.google.com/uc?id=1267Q7843pq-1pUsIsLFBBkwhJw9-QhVl",
    "https://drive.google.com/uc?id=19kJ5olxKirqvPFQ6yp6_cHZpab7QPoSM",
    "https://drive.google.com/uc?id=130HXZqh2gfe3t00zGcncGOdjvDmHVpDF",
    "https://drive.google.com/uc?id=196tG7VB3kSh9s38DNuYU86tCImORbKsR",
    "https://drive.google.com/uc?id=11RN1lEMGmIOxY3NqEpidNHwTGIOq2no_",
    "https://drive.google.com/uc?id=17CmU34Hu5jjOz3EOfO6pzreINys4CLvF",
    "https://drive.google.com/uc?id=14pazYA3MwMzd2XJ394xzB5JTm3rVkw6k",
    "https://drive.google.com/uc?id=18iRTXvB91ZqVS25FDcRmWhOS3ZTGCZuh",
    "https://drive.google.com/uc?id=1NboxPDmgKoqGjoPt4mhgWo-thtTwVLiK",
    "https://drive.google.com/uc?id=1M-ttSt_hPU4mu0HROfevMvVF8PhRn9P7",
    "https://drive.google.com/uc?id=1_eWVtiqojvvl-l8m-m41hlr-bkEFbolz",
    "https://drive.google.com/uc?id=1TcxQYfD3DeJkcmyaJur-w9oxgNbIVxWg",
    "https://drive.google.com/uc?id=1udhFpy_5iw2f-q4xtL4ygGKb1o92_o-v",
    "https://drive.google.com/uc?id=11sOmPQVQDZkMl0OHw0sWNRMUCWHdLmf-",
    "https://drive.google.com/uc?id=1UTYKpb4ffDGp2T5p7m8yXTShRBFeLfI-",
    "https://drive.google.com/uc?id=1muksmgicTCZ0efpd_v3uM9i2Os2d44Jb",
    "https://drive.google.com/uc?id=1mug4ejZENj8k45D2nuB1oGswygG3UBDg",
    "https://drive.google.com/uc?id=1rwkGlmY8KrcXc-bvbA3gUmdSAtu7JjzN",
    "https://drive.google.com/uc?id=1UqighGwemTaGVorYagaBIGQ3c0vxwQfm",
    "https://drive.google.com/uc?id=1xV0tMgV1Mq5Q50HjUz-Wftv-fCUOpA-i",
    "https://drive.google.com/uc?id=1UTx-EdWHiiJc14btxudQOzqP9ZpqzU-D",
    "https://drive.google.com/uc?id=1LCLyt09v0Kw0rMFjzFYFtfBvq18c5bVf",
    "https://drive.google.com/uc?id=1Ig-Fcw3MSyjpNwjbxZQj_UlIGg8qlSEV",
    "https://drive.google.com/uc?id=18DYNbS9RciD7MRSWuYyWqvwwsKvIvfv1",
    "https://drive.google.com/uc?id=1itjpNG_IMwRT-E5OvqejY_XqBm8PuqIS",
    "https://drive.google.com/uc?id=136a9uhPVlORt0ykxIsnrMPNph-zM2wvo",
    "https://drive.google.com/uc?id=1KnbGszfwALFyMBZL5nckeXn05ow9bvzv",
    "https://drive.google.com/uc?id=1-IKOa-n2X1bfIQrcmoEpgFvghXDdXiuN",
    "https://drive.google.com/uc?id=1CDg07AVi8TkicrH3HYXoec-GBr_ByTit",
    "https://drive.google.com/uc?id=11isbUC6egQcsygbUiNzhzRWou9C3j5_E",
    "https://drive.google.com/uc?id=1tmLF1SxoqdKDYrRCY_HsvqPULfO4mPjS",
    "https://drive.google.com/uc?id=1L6-VzdaLqv1dbccUdA-YyGX48GqDVmtF",
    "https://drive.google.com/uc?id=1GmVqk5KUOnFD5m6f4Q-Db-CWXHK7uB7w",
    "https://drive.google.com/uc?id=1-cNnSzQ_lk1MDtOlBdZX7H6nBm6XWOAi",
    "https://drive.google.com/uc?id=1HDtkK_IZ82ot1IpJHMjsnoD6kNF3Etgu",
    "https://drive.google.com/uc?id=1eQUzmbQJjJ_6EVEHPKO42E_JDZdAtdri",
    "https://drive.google.com/uc?id=1Wxl1r0JcmDMMGlucVd8SXiCTE5-yk6QU",
    "https://drive.google.com/uc?id=1N0nVBbgLcWVhNKlcxlhpinY3M_IQlndi",
    "https://drive.google.com/uc?id=1AyPPGUAjrc8xQJSQfWyFG2cmvkNRSt15",
    "https://drive.google.com/uc?id=1eRZ8PBe0oo0FtMdR0M6hibCSLwRvhWBT",
    "https://drive.google.com/uc?id=12emTPShQm1-rn9wweT3cQwK_-4cU_GGx",
    "https://drive.google.com/uc?id=1d-wVxSQ7LExQmkIjFnuXjJF8I4HLLhzs",
    "https://drive.google.com/uc?id=1QvAT8OefOePHDMnMHrpNG6_-VK8pIXWr",
    "https://drive.google.com/uc?id=16Tccnj-qCI5y_MyrCFnOXrzeELBXiczW",
    "https://drive.google.com/uc?id=1Dm_t1V3H5PHrYLLBA4XKL2FMVBcggT-S",
    "https://drive.google.com/uc?id=1R-30KC4MtJz-H1eOZHKN52TPvEwk64AP",
    "https://drive.google.com/uc?id=1JNLjuzMmJeRQMlUhD4KVsADhVUFmvkuB",
    "https://drive.google.com/uc?id=1UXZmwYg7Z_PRSF-gmr7AKoK0AsSMY1HF",
    "https://drive.google.com/uc?id=1M8JPwyCRuRralMR4TdiEieR51vn5M7SO",
    "https://drive.google.com/uc?id=11SPvbuO9YwFjDsjH6KFAUc0wOIkHgiyn",
    "https://drive.google.com/uc?id=1yJe_AxI7Jn1DQGzyZEpOTErxz896puMP",
    "https://drive.google.com/uc?id=127BASMNMCjuX8-_yf3soQp38O9kZsSJT",
    "https://drive.google.com/uc?id=1R_Qk4odgBDNmp9ksZyHLmUl94GnpwFER",
    "https://drive.google.com/uc?id=1Fhn_pn3RkG2qFpjWrnC6zNWA4uzN54nu",
    "https://drive.google.com/uc?id=1qZjhHV4v_6pSBMPP5wAjCAFfKuRnzuCL",
    "https://drive.google.com/uc?id=1sacfcvPOgxLT3rCdvziWXOJzYWcApABK",
    "https://drive.google.com/uc?id=1FIzaPgb1RJgErnTOJIO_PwtGt9Y-KDuh",
    "https://drive.google.com/uc?id=1vPoshhqGeiB-Z_vtJhdw1opbcz3yaJH0",
    "https://drive.google.com/uc?id=1WoS1-6uR29tauRuQdFWdiUUUapMpOa9M",
    "https://drive.google.com/uc?id=1-utro1jK6rySS9ZSjjUx86Bsnmem0g3F",
    "https://drive.google.com/uc?id=13CPRo01vZtaS8MK2U2eKYIRKvcXg_fo9",
    "https://drive.google.com/uc?id=14bKhiTnWqBF0QYbHCwpGFefCGwXrmxW_",
    "https://drive.google.com/uc?id=1T7R0WlNkfeTjFeRsvpz6GBh-E4rO6gKJ",
    "https://drive.google.com/uc?id=1iA-8jBjs2cl0Ux9VgqjMfTXdCCCiW2nZ",
    "https://drive.google.com/uc?id=1Vf33sYIK3B3mPstQHvdYwJbBCgLwtZPS",
    "https://drive.google.com/uc?id=1D7RL116hozVthsh-kH0jyLtzMWRFM0RY",
    "https://drive.google.com/uc?id=1RIv8QtS393S3bOrA2gRpuZ1UrTu2-7vB",
    "https://drive.google.com/uc?id=1QxXmOSBO5gHQSjky-XLkVdZJTMwDENRX",
    "https://drive.google.com/uc?id=1ytCq4dv8-U0ndDwH7YFCb0cPzzoc2Lo5",
    "https://drive.google.com/uc?id=1ENrGOCmUO4ASSOOQ4RLF2wS5dqtc3SSP",
    "https://drive.google.com/uc?id=1ti4TBYNgqYDSYru953KQSNSphvE2kupb",
    "https://drive.google.com/uc?id=1ymi8E0SOo8J6GZ5BlBtckhwetHnIq7fU",
    "https://drive.google.com/uc?id=1bdtNPEmV-9DikMMEsUP4aGwhHFdr3DP_",
    "https://drive.google.com/uc?id=1ExTfS3LbE9KHl-brGHSh7LTEoroWyhxS",
    "https://drive.google.com/uc?id=1OYrDMQ6UxjFVrDIRu3-C25LKad_Dmvj6",
    "https://drive.google.com/uc?id=13LgtVOyeD7jTPNsgLWRBOfLELtI-KDxo",
    "https://drive.google.com/uc?id=1xTPy2oiydcntRsRBQ4EfvoCcagiav5Um",
    "https://drive.google.com/uc?id=1CIssf7ov5YD1hWDFPhdys4Uvxh6pD2qE",
    "https://drive.google.com/uc?id=1ZJtCyljtF47Jg4AtdZWecLalPiEk5f3W",
    "https://drive.google.com/uc?id=1YZUs3CpYnQ5rdXlwMtcwXeHoUFjPa2s1",
    "https://drive.google.com/uc?id=11J1DRr-jlUIgDnLCPRRj1DQv-03XgArp",
    "https://drive.google.com/uc?id=1cZKnVzI5Wh8Utjqt9U4hVLKpDWCM59E_",
    "https://drive.google.com/uc?id=19x3fTZJCYr_faRgDrPI-wM2n4nBfH-uE",
    "https://drive.google.com/uc?id=1UBOgoMcTEipS4-ivRQuMjGFqnxTHaka-",
    "https://drive.google.com/uc?id=1hg8dZXuEaBQx16A7YxiYYuu4_CvzFU8t",
    "https://drive.google.com/uc?id=1PiRu4gaR7L9GlwTw5X_TrVDlieyIxptm",
    "https://drive.google.com/uc?id=1esFJD7PFasNWs2CcNo0UVnDf3Kxs-AHr",
    "https://drive.google.com/uc?id=1EKmLcv2MIRYBg4hZiCguL0knBOudr8sV",
    "https://drive.google.com/uc?id=1JrXtsd3lSbpP3Zf12lveFP4t9gaeS6n4",
    "https://drive.google.com/uc?id=1m7hmnmp1BOOPRdi83dE06plW2xzOLkiJ",
    "https://drive.google.com/uc?id=1HHpqOWNHOkb4f7waWSX8EkQzJDZFywh_",
    "https://drive.google.com/uc?id=16z7WSKz5C_inlZ72Oy5FUenkyoIC0VOY",
    "https://drive.google.com/uc?id=1_bor67sTlBzP9VcXpIV9OHbViU07xjhk",
    "https://drive.google.com/uc?id=1CKs61TKdaIIGYGw0HxwJyYDUTQjzLfDl",
    "https://drive.google.com/uc?id=1oqVOUqckl-m9wR_SRh0OmwsqrB6w1QWg",
    "https://drive.google.com/uc?id=1MsugMjnIbmJ5wABjmuPPuL3cJ4JyrMuE",
    "https://drive.google.com/uc?id=1HgagtBKb5IDL_D9wa9kZ8TZnnGPl1OvG",
    "https://drive.google.com/uc?id=1dPH2uEK1EWT-plttOne_TpPga0gqsDlb",
    "https://drive.google.com/uc?id=1sAT8_TdhJmkOFqM6dnX3bWUinHhqJGz4",
    "https://drive.google.com/uc?id=17ZQ_F5iO6cdaxPPEbtld6Gv1O_4oXLxx",
    "https://drive.google.com/uc?id=1ljr6IR0X7YHM97DKrXCxzzGN5Z8IIpHR",
    "https://drive.google.com/uc?id=1HfDjVgkFPAUhQ5i_LrYnQ60_PCAY1KDP",
    "https://drive.google.com/uc?id=1bJ75tUf-ERMIifoUk4kxXg5Ft9bBM0LH",
    "https://drive.google.com/uc?id=1LUrjXwtxtrelTrFalUnP0ns7EYWAg7ys",
    "https://drive.google.com/uc?id=1DGNeTbtAuXpf8D8y414KkLnppXQK8WZ2",
    ]

class universemachine_catalogs(sim_catalogs):
    def __init__(self,
                 sim, path='auto',
                 snaps='all',
                 updatepath=True,
                 ):
        super().__init__(sim, path, snaps, updatepath)


        # Identify keys that will require unit conversions
        # For UM these are empty because everythings in units
        # we like already.
        self.mass_keys = []
        self.mass_keys_add_h = ['sm','icl','obs_sm']
        self.pos_keys = ['r']
        self.inv_time_keys = []

        self.basic_fields = {
            #Flags: Mostly internal UniverseMachine info.  However, halos with bit 4 set in the flags (i.e., flags & (2**4) is true) should be ignored.
            'flags':[('flag','i4','none',0)],

            #pos[6]: (X,Y,Z,VX,VY,VZ)
            #X Y Z: halo position (comoving Mpc/h)
            #VX VY VZ: halo velocity (physical peculiar km/s)
            'pos':[('pos_x','f4','Mpc/h',-1),('pos_y','f4','Mpc/h',-1),('pos_z','f4','Mpc/h',-1),
                   ('v_x','f4','km/s',0),('v_y','f4','km/s',0),('v_z','f4','km/s',0)],

            #M: Halo mass (Bryan & Norman 1998 virial mass, Msun/h)
            'm':[('mass','f4','Msun/h',-1)],
            }

        self.dm_fields = {
            #UPID: -1 for central halos, otherwise, ID of largest parent halo
            'upid':[('parent_id','i8','None',0)],

            #V: Halo vmax (physical km/s)
            'v':[('vmax','f4','km/s',0)],

            #MP: Halo peak historical mass (BN98 vir, Msun/h)
            'mp':[('mass_peak','f4','Msun/h',-1)],

            #VMP: Halo vmax at the time when peak mass was reached.
            'vmp':[('vmax_peak','f4','km/s',0)],

            #R: Halo radius (BN98 vir, comoving kpc/h)
            'r':[('r','f4','kpc/h',0)],
            }

        self.matter_fields = {
            #A_UV: UV attenuation (mag)
            'A_UV':[('uv_attenuation','f4','mag',0)],

            #SM: True stellar mass (Msun)
            'sm':[('m_stars','f4','Msun/h',-1)],

            #ICL: True intracluster stellar mass (Msun)
            'icl':[('icl','f4','Msun/h',-1)],

            #SFR: True star formation rate (Msun/yr)
            'sfr':[('sfr','f4','Msun/yr',0)],

            #Obs_SM: observed stellar mass, including random & systematic errors (Msun)
            'obs_sm':[('m_stars_obs','f4','Msun/h',-1)],

            #Obs_SFR: observed SFR, including random & systematic errors (Msun/yr)
            'obs_sfr':[('sfr_obs','f4','Msun/yr',0)],

            #Obs_UV: Observed UV Magnitude (M_1500 AB)
            'obs_uv':[('phot_uv_obs','f4','mag',0)],
            }

        # Some stuff specific to each simulation module
        if self.sim == 'UniverseMachine-SMDPL':
            self.a = a_smdpl
            self.urls = links_smdpl

        # Check whether snapshots have been downloaded
        not_downloaded = []
        for snap in self.snaps:
            file_path = self.path+'/raw/sfr_catalog_{}.bin'.format(self.a[snap])
            if not os.path.exists(file_path):
                not_downloaded.append(snap)
        if len(not_downloaded) > 0:
            warnings.warn("No data exists for snapshots {} - run .download".format(not_downloaded))

        # Function to load the data in a format we want:
        dtype_raw = np.dtype(dtype=[('id', 'i8'),('descid','i8'),('upid','i8'),
                                    ('flags', 'i4'), ('uparent_dist', 'f4'),
                                    ('pos', 'f4', (6)), ('vmp', 'f4'), ('lvmp', 'f4'),
                                    ('mp', 'f4'), ('m', 'f4'), ('v', 'f4'), ('r', 'f4'),
                                    ('rank1', 'f4'), ('rank2', 'f4'), ('ra', 'f4'),
                                    ('rarank', 'f4'), ('A_UV', 'f4'), ('sm', 'f4'),
                                    ('icl', 'f4'), ('sfr', 'i4'), ('obs_sm', 'f4'),
                                    ('obs_sfr', 'f4'), ('obs_uv', 'f4'), ('empty', 'f4')],
                             align=True)
        def loader(path, snapshot, fields):

            flag = 16
            a_str = self.a[snapshot]
            data_raw = np.fromfile(path+'/sfr_catalog_{}.bin'.format(a_str), dtype=dtype_raw)
            data_raw = data_raw[(data_raw['flags'] & flag) != 16]

            subhalos = {}
            for key in {**self.basic_fields,**self.dm_fields,**self.matter_fields}.keys():
                subhalos[key] = data_raw[key]
            subhalos['pos'][:,0:3] % self.box_edge
            n_halos = len(data_raw)

            return subhalos, n_halos

        self.loader = loader

    def download_meta(self, remake=False):

        # Check that metadata doesn't already exist
        if not remake:
            if os.path.exists(self.meta_path):
                warnings.warn("Metadata appears to exist already")
                return

        # Assign the correct data for different simulation boxes
        if self.sim == 'UniverseMachine-SMDPL':
            self.metadata = {'name':self.sim,
                             'box_edge':400,
                             'number_snaps':n_snap_smdpl
                             }
        self.metadata['cosmo_name'] = 'Planck'
        self.metadata['cosmo_omega_matter'] = 0.307
        self.metadata['cosmo_omega_lambda'] = 0.693
        self.metadata['cosmo_omega_baryon'] = 0.048
        self.metadata['cosmo_h'] = 0.678

        self.box_edge = self.metadata['box_edge']
        self.h = self.metadata['cosmo_h']

        # Snapshots for different simulation boxes
        numbers = range(len(self.a))

        snap_meta_classes = []
        for i in range(self.metadata['number_snaps']):
            redshift = 1/float(self.a[i])-1
            snap_meta_classes.append(snapshot(numbers[i],redshift,self.metadata))

        snap_meta_classes = [snap_meta_classes[i] for i in np.argsort(numbers)]
        numbers = np.sort(numbers)

        snap_meta_classes = [snap_meta_classes[i] for i in numbers if i in self.snaps]

        snap_meta_classes[0].dif_higherz_snap('max')
        snap_meta_classes[-1].dif_lowerz_snap(0)
        for i in range(len(snap_meta_classes)-1):
            snap_meta_classes[i].dif_lowerz_snap(snap_meta_classes[i+1])
            snap_meta_classes[i+1].dif_higherz_snap(snap_meta_classes[i])

        snap_meta_dtype = [('index','i'),
                           ('redshift','f'),
                           ('redshift_min','f'),('redshift_max','f'),
                           ('time_start','f'),('time_end','f'),
                           ('distance_min','f'),('distance_max','f'),
                           ('transverse_distance_min','f'),('transverse_distance_max','f')]
        snap_meta = np.zeros(len(snap_meta_classes),
                             dtype = snap_meta_dtype)

        for i in range(len(snap_meta_classes)):
            snap_meta[i]['index'] = snap_meta_classes[i].index
            snap_meta[i]['redshift'] = snap_meta_classes[i].redshift
            snap_meta[i]['redshift_min'] = snap_meta_classes[i].redshift_min
            snap_meta[i]['redshift_max'] = snap_meta_classes[i].redshift_max
            snap_meta[i]['time_start'] = snap_meta_classes[i].time_start *  self.h
            snap_meta[i]['time_end'] = snap_meta_classes[i].time_end *  self.h
            snap_meta[i]['distance_min'] = snap_meta_classes[i].distance_min *  self.h
            snap_meta[i]['distance_max'] = snap_meta_classes[i].distance_max *  self.h
            snap_meta[i]['transverse_distance_min'] = snap_meta_classes[i].transverse_distance_min *  self.h
            snap_meta[i]['transverse_distance_max'] = snap_meta_classes[i].transverse_distance_max *  self.h
        self.snap_meta = snap_meta

        np.save(self.meta_path,self.metadata)
        np.save(self.snap_meta_path,self.snap_meta)

    def download(self, redownload=False):

        if not os.path.exists(self.path+'/raw'):
            os.mkdir(self.path+'/raw')

        # Add a check for already downloaded files
        if redownload:
            self.download_snaps = np.copy(self.snaps)
        else:
            self.download_snaps = []
            for snap in self.snaps:
                file_path = self.path+'/raw/sfr_catalog_{}.bin'.format(self.a[snap])
                if os.path.exists(file_path):
                    warnings.warn("Skipping snapshot {} as it appears to exist already".format(snap))
                else:
                    self.download_snaps.append(snap)

        # Download each snap
        for i in range(len(self.download_snaps)):
            snap = self.download_snaps[i]
            url = self.urls[snap]
            print("\nDownloading item {} of {}".format(i+1,len(self.download_snaps)))
            gdown.download(url, output=self.path+'/raw/')
