class SQLQueries:
    """Contains SQL queries for MIMIC-III database"""
    
    @staticmethod
    def get_admissions_query(limit=None):
        """Query for admissions data"""
        query = """
        SELECT 
            subject_id,
            hadm_id,
            admission_type,
            admission_location,
            discharge_location,
            insurance,
            language,
            religion,
            marital_status,
            ethnicity,
            diagnosis,
            hospital_expire_flag,
            DATE_PART('year', age(admittime, dob)) as admission_age,
            EXTRACT(EPOCH FROM (dischtime - admittime))/3600 as length_of_stay_hours
        FROM admissions a
        JOIN patients p ON a.subject_id = p.subject_id
        """
        if limit:
            query += f" LIMIT {limit}"
        return query
    
    @staticmethod
    def get_patients_query(limit=None):
        """Query for patient demographics"""
        query = """
        SELECT 
            subject_id,
            gender,
            dob,
            EXTRACT(YEAR FROM dob) as birth_year,
            CASE 
                WHEN gender = 'M' THEN 1
                WHEN gender = 'F' THEN 0
                ELSE NULL 
            END as gender_numeric,
            EXTRACT(YEAR FROM age(CURRENT_DATE, dob)) as current_age
        FROM patients
        """
        if limit:
            query += f" LIMIT {limit}"
        return query
    
    @staticmethod
    def get_icustays_query(limit=None):
        """Query for ICU stays"""
        query = """
        SELECT 
            subject_id,
            hadm_id,
            icustay_id,
            first_careunit,
            last_careunit,
            first_wardid,
            last_wardid,
            EXTRACT(EPOCH FROM (outtime - intime))/3600 as icu_los_hours
        FROM icustays
        WHERE outtime IS NOT NULL
        """
        if limit:
            query += f" LIMIT {limit}"
        return query
    
    @staticmethod
    def get_diagnoses_query(limit=None):
        """Query for diagnoses"""
        query = """
        SELECT 
            subject_id,
            hadm_id,
            icd9_code,
            seq_num
        FROM diagnoses_icd
        WHERE icd9_code IS NOT NULL
        """
        if limit:
            query += f" LIMIT {limit}"
        return query
    
    @staticmethod
    def get_vitals_query(limit=None):
        """Query for vital signs"""
        query = """
        SELECT 
            ce.subject_id,
            ce.hadm_id,
            ce.icustay_id,
            ce.itemid,
            di.label as measurement,
            ce.charttime,
            ce.valuenum as value,
            ce.valueuom as unit
        FROM chartevents ce
        JOIN d_items di ON ce.itemid = di.itemid
        WHERE ce.valuenum IS NOT NULL 
            AND ce.valuenum > 0
            AND di.category IN ('Respiratory', 'Vital Signs', 'Cardiovascular')
        """
        if limit:
            query += f" LIMIT {limit}"
        return query
    
    @staticmethod
    def get_labs_query(limit=None):
        """Query for laboratory results"""
        query = """
        SELECT 
            subject_id,
            hadm_id,
            itemid,
            charttime,
            valuenum as value,
            valueuom as unit,
            flag
        FROM labevents
        WHERE valuenum IS NOT NULL
            AND valuenum > 0
            AND itemid IN (
                SELECT itemid FROM d_labitems 
                WHERE category IN ('Chemistry', 'Hematology', 'Blood Gas')
            )
        """
        if limit:
            query += f" LIMIT {limit}"
        return query
    
    @staticmethod
    def get_notes_query(limit=None):
        """Query for clinical notes"""
        query = """
        SELECT 
            subject_id,
            hadm_id,
            chartdate,
            category,
            description,
            text
        FROM noteevents
        WHERE iserror IS NULL 
            OR iserror != '1'
            AND text IS NOT NULL
            AND LENGTH(text) > 100
        """
        if limit:
            query += f" LIMIT {limit}"
        return query
    
    @staticmethod
    def get_patient_full_data(subject_id):
        """Get complete patient data for prediction"""
        query = f"""
        WITH patient_info AS (
            SELECT 
                p.subject_id,
                p.gender,
                DATE_PART('year', age(CURRENT_DATE, p.dob)) as age,
                a.hadm_id,
                a.admission_type,
                a.ethnicity,
                a.diagnosis,
                i.icustay_id,
                i.first_careunit,
                i.last_careunit
            FROM patients p
            LEFT JOIN admissions a ON p.subject_id = a.subject_id
            LEFT JOIN icustays i ON a.hadm_id = i.hadm_id
            WHERE p.subject_id = {subject_id}
            ORDER BY a.admittime DESC
            LIMIT 1
        ),
        latest_vitals AS (
            SELECT 
                ce.subject_id,
                MAX(CASE WHEN di.label ILIKE '%heart rate%' THEN ce.valuenum END) as heart_rate,
                MAX(CASE WHEN di.label ILIKE '%blood pressure%' AND di.label ILIKE '%systolic%' THEN ce.valuenum END) as systolic_bp,
                MAX(CASE WHEN di.label ILIKE '%blood pressure%' AND di.label ILIKE '%diastolic%' THEN ce.valuenum END) as diastolic_bp,
                MAX(CASE WHEN di.label ILIKE '%temperature%' THEN ce.valuenum END) as temperature,
                MAX(CASE WHEN di.label ILIKE '%respiratory rate%' THEN ce.valuenum END) as respiratory_rate,
                MAX(CASE WHEN di.label ILIKE '%oxygen saturation%' THEN ce.valuenum END) as spo2
            FROM chartevents ce
            JOIN d_items di ON ce.itemid = di.itemid
            WHERE ce.subject_id = {subject_id}
                AND ce.valuenum IS NOT NULL
                AND ce.valuenum > 0
            GROUP BY ce.subject_id
        ),
        latest_labs AS (
            SELECT 
                le.subject_id,
                MAX(CASE WHEN dli.label ILIKE '%white blood cell%' THEN le.valuenum END) as wbc,
                MAX(CASE WHEN dli.label ILIKE '%lactate%' THEN le.valuenum END) as lactate,
                MAX(CASE WHEN dli.label ILIKE '%creatinine%' THEN le.valuenum END) as creatinine,
                MAX(CASE WHEN dli.label ILIKE '%platelet%' THEN le.valuenum END) as platelets,
                MAX(CASE WHEN dli.label ILIKE '%bilirubin%' THEN le.valuenum END) as bilirubin
            FROM labevents le
            JOIN d_labitems dli ON le.itemid = dli.itemid
            WHERE le.subject_id = {subject_id}
                AND le.valuenum IS NOT NULL
                AND le.valuenum > 0
            GROUP BY le.subject_id
        ),
        sepsis_diagnosis AS (
            SELECT 
                subject_id,
                hadm_id,
                CASE WHEN EXISTS (
                    SELECT 1 FROM diagnoses_icd d 
                    WHERE d.subject_id = a.subject_id 
                    AND d.hadm_id = a.hadm_id
                    AND d.icd9_code IN ('038', '785.52', '995.91', '995.92')
                ) THEN 1 ELSE 0 END as has_sepsis
            FROM admissions a
            WHERE a.subject_id = {subject_id}
        )
        SELECT 
            pi.*,
            lv.heart_rate,
            lv.systolic_bp,
            lv.diastolic_bp,
            lv.temperature,
            lv.respiratory_rate,
            lv.spo2,
            ll.wbc,
            ll.lactate,
            ll.creatinine,
            ll.platelets,
            ll.bilirubin,
            sd.has_sepsis
        FROM patient_info pi
        LEFT JOIN latest_vitals lv ON pi.subject_id = lv.subject_id
        LEFT JOIN latest_labs ll ON pi.subject_id = ll.subject_id
        LEFT JOIN sepsis_diagnosis sd ON pi.subject_id = sd.subject_id AND pi.hadm_id = sd.hadm_id
        """
        return query