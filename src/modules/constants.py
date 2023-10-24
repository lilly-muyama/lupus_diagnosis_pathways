class Constants:
    def init(self, args):
        self.SEED = args.seed
        self.LAMBDA = args.lambda_constant
        self.CORRECT_DIAGNOSIS_REWARD = 1
        self.INCORRECT_DIAGNOSIS_REWARD = -1
        self.REPEATED_ACTION_REWARD = -1
        self.MAX_LENGTH_REWARD = -1
        self.FIRST_ACTION_REWARD = 0

        self.ACTION_SPACE = ['No lupus', 'Lupus', 'Inconclusive diagnosis', 'ana', 'fever', 'leukopenia', 'thrombocytopenia', 'auto_immune_hemolysis', 'delirium',
        'psychosis', 'seizure', 'non_scarring_alopecia', 'oral_ulcers', 'cutaneous_lupus', 'pleural_effusion', 'pericardial_effusion', 'acute_pericarditis',
        'joint_involvement', 'proteinuria', 'biopsy_proven_lupus_nephritis', 'anti_cardioliphin_antibodies', 'anti_β2gp1_antibodies', 'lupus_anti_coagulant', 'low_c3', 
        'low_c4', 'anti_dsdna_antibody', 'anti_smith_antibody']

        self.ACTION_NUM = len(self.ACTION_SPACE)

        self.CLASS_DICT = {'No lupus':0, 'Lupus':1, 'Inconclusive diagnosis':2} 

        self.CLASS_NUM = len(self.CLASS_DICT)

        self.FEATURE_NUM = self.ACTION_NUM - self.CLASS_NUM

        self.MAX_STEPS = self.FEATURE_NUM + 1

        self.CHECKPOINT_FREQ = 1000000

        self.DOMAINS_FEAT_DICT = {'constitutional': ['fever'],
        'hematologic': ['leukopenia', 'thrombocytopenia', 'auto_immune_hemolysis'],
        'neuropsychiatric': ['delirium', 'psychosis', 'seizure'],
        'mucocutaneous': ['non_scarring_alopecia', 'oral_ulcers', 'cutaneous_lupus'],
        'serosal': ['pleural_effusion', 'pericardial_effusion', 'acute_pericarditis'],
        'musculoskeletal': ['joint_involvement'],
        'renal': ['proteinuria', 'biopsy_proven_lupus_nephritis'],
        'antiphospholipid_antibodies': ['anti_cardioliphin_antibodies', 'anti_β2gp1_antibodies', 'lupus_anti_coagulant'],
        'complement_proteins': ['low_c3', 'low_c4'],
        'sle_specific_antibodies':['anti_dsdna_antibody', 'anti_smith_antibody']}


        self.DOMAINS_MAX_SCORES_DICT = {'constitutional': 2, 'hematologic': 4, 'neuropsychiatric': 5, 'mucocutaneous': 6, 'serosal': 6, 'musculoskeletal': 6, 
        'renal': 10, 'antiphospholipid_antibodies': 2, 'complement_proteins': 4, 'sle_specific_antibodies':6}

        self.CRITERIA_WEIGHTS = {'fever':2, 'leukopenia':3, 'thrombocytopenia':4, 'auto_immune_hemolysis':4, 'delirium':2, 'psychosis':3, 'seizure':5, 
        'non_scarring_alopecia':2, 'oral_ulcers':2, 'subacute_cutaneous_lupus':4, 'discoid_lupus':4, 'acute_cutaneous_lupus':6, 'pleural_effusion': 5, 
        'pericardial_effusion':5, 'acute_pericarditis':6, 'joint_involvement':6, 'proteinuria':4, 'renal_biopsy_1':0, 'renal_biopsy_2':8, 'renal_biopsy_3':10, 
        'renal_biopsy_4':10, 'renal_biopsy_5':8, 'anti_cardioliphin_antibodies':2, 'anti_β2gp1_antibodies':2, 'lupus_anti_coagulant':2, 'low_c3':3, 'low_c4':3, 
        'low_c3_and_low_c4':4, 'low_c3_or_low_c4':3, 'anti_dsdna_antibody':6, 'anti_smith_antibody':6}

        self.FEATURE_SCORES = {'ana':11.5, 'fever':15, 'leukopenia':12, 'thrombocytopenia':12, 'auto_immune_hemolysis':12, 'delirium':15, 'psychosis':15, 'seizure':15, 
        'non_scarring_alopecia':15, 'oral_ulcers':15, 'cutaneous_lupus':6.5, 'pleural_effusion':13, 'pericardial_effusion':13, 'acute_pericarditis':13, 
        'joint_involvement':13, 'proteinuria':13.5, 'biopsy_proven_lupus_nephritis':1, 'anti_cardioliphin_antibodies':10, 'anti_β2gp1_antibodies':9.5, 
        'lupus_anti_coagulant':9.5, 'low_c3':10.5, 'low_c4':10.5, 'anti_dsdna_antibody':11.5, 'anti_smith_antibody':11.5}

        self.MAX_FEATURE_SCORE = 283.5
        self.REG_FEATURE_SCORE = 0.9

constants = Constants()