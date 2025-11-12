SELECT "FK_episode_id",
        "patientID",
        "startDTS",
        eps."admitPatientEncounterID",
        eps."dcPatientEncounterID"
    FROM {{cohort_table}} as esc
    LEFT JOIN {{episode_table}} AS eps
        ON esc."FK_episode_id" = eps."id"