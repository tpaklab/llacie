CREATE TABLE IF NOT EXISTS "{{episode_table}}" (
    "id"                        BIGSERIAL PRIMARY KEY,
    "patientID"                 VARCHAR(40) NOT NULL,
    "admitPatientEncounterID"   BIGINT NOT NULL,
    "dcPatientEncounterID"      BIGINT NOT NULL,
    "startDTS"                  TIMESTAMP WITH TIME ZONE
);

CREATE TABLE IF NOT EXISTS "{{cohort_table}}" (
    "id"                        BIGSERIAL PRIMARY KEY,
    "FK_episode_id"             BIGINT NOT NULL,
    "infectionCriteria"         BOOLEAN NOT NULL DEFAULT TRUE,
    "excl_ST0_combined"         BOOLEAN NOT NULL DEFAULT FALSE
);

ALTER TABLE "{{cohort_table}}" 
    ADD CONSTRAINT "{{cohort_table}}_FK_episode_id_fk" 
    FOREIGN KEY ("FK_episode_id")
    REFERENCES "{{episode_table}}"("id")
    ON DELETE RESTRICT
    ON UPDATE RESTRICT;

CREATE INDEX IF NOT EXISTS "{{cohort_table}}_FK_episode_id_key"
    ON "{{cohort_table}}" ("FK_episode_id");

CREATE TABLE "{{prefix}}notes" (
    "id"                    BIGSERIAL PRIMARY KEY,
    "FK_episode_id"         BIGINT NOT NULL,
    "NoteID"                BIGINT UNIQUE NOT NULL,
    "PatientEncounterID"    BIGINT NOT NULL,
    "InpatientNoteTypeDSC"  VARCHAR(255) NOT NULL DEFAULT 'H&P',
    "CurrentAuthorID"       VARCHAR(30),
    "AuthorServiceDSC"      VARCHAR(255),
    "AuthorProviderTypeDSC" VARCHAR(255),
    "CreateDTS"             TIMESTAMP WITH TIME ZONE,
    "DateOfServiceDTS"      TIMESTAMP WITH TIME ZONE,
    "LastFiledDTS"          TIMESTAMP WITH TIME ZONE,
    "EDWLastModifiedDTS"    TIMESTAMP WITH TIME ZONE,
    "NoteFiledDTS"          TIMESTAMP WITH TIME ZONE,
    "ContactDateRealNBR"    NUMERIC(8, 2),
    "NoteCSNID"             BIGINT,
    "note_text"             TEXT
);

ALTER TABLE "{{prefix}}notes" 
    ADD CONSTRAINT "{{prefix}}notes_FK_episode_id_fk" 
    FOREIGN KEY ("FK_episode_id")
    REFERENCES "{{episode_table}}"("id")
    ON DELETE RESTRICT
    ON UPDATE RESTRICT;

CREATE INDEX IF NOT EXISTS "{{prefix}}notes_FK_episode_id_key"
    ON "{{prefix}}notes" ("FK_episode_id");

CREATE TABLE "{{prefix}}tasks" (
    "id"                    BIGSERIAL PRIMARY KEY,
    "output_type"           VARCHAR(255) NOT NULL,
    "name"                  VARCHAR(255) NOT NULL,
    "desc"                  TEXT
);

CREATE TABLE "{{prefix}}strategies" (
    "id"                    BIGSERIAL PRIMARY KEY,
    "FK_task_id"            BIGINT NOT NULL,
    "name"                  VARCHAR(255) NOT NULL,
    "desc"                  TEXT,
    "version"               VARCHAR(255) NOT NULL,
    "last_updated"          TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

ALTER TABLE "{{prefix}}strategies" 
    ADD CONSTRAINT "{{prefix}}strategies_FK_task_id_fk" 
    FOREIGN KEY ("FK_task_id")
    REFERENCES "{{prefix}}tasks"("id")
    ON DELETE RESTRICT
    ON UPDATE RESTRICT;

CREATE TABLE "{{prefix}}note_sections" (
    "id"                    BIGSERIAL PRIMARY KEY,
    "FK_note_id"            BIGINT NOT NULL,
    "section_name"          VARCHAR(40) NOT NULL,
    "section_value"         TEXT,
    "FK_strategy_id"        BIGINT NOT NULL,
    "last_updated"          TIMESTAMP WITH TIME ZONE
);

ALTER TABLE "{{prefix}}note_sections" 
    ADD CONSTRAINT "{{prefix}}note_sections_FK_note_id_fk" 
    FOREIGN KEY ("FK_note_id")
    REFERENCES "{{prefix}}notes"("id")
    ON DELETE RESTRICT
    ON UPDATE RESTRICT;

ALTER TABLE "{{prefix}}note_sections" 
    ADD CONSTRAINT "{{prefix}}note_sections_FK_strategy_id_fk" 
    FOREIGN KEY ("FK_strategy_id")
    REFERENCES "{{prefix}}strategies"("id")
    ON DELETE RESTRICT
    ON UPDATE RESTRICT;

CREATE TABLE "{{prefix}}note_features" (
    "id"                    BIGSERIAL PRIMARY KEY,
    "FK_note_id"            BIGINT NOT NULL,
    "feature_name"          VARCHAR(255) NOT NULL,
    "FK_note_section_id"    BIGINT,
    "FK_strategy_id"        BIGINT NOT NULL,
    "llm_output_raw"        TEXT NOT NULL,
    "feature_value"         TEXT,
    "feature_updated"       TIMESTAMP WITH TIME ZONE NOT NULL,
    "strategy_runtime"      NUMERIC(12, 5)
);

ALTER TABLE "{{prefix}}note_features" 
    ADD CONSTRAINT "{{prefix}}note_features_note_feat_strategy_key" 
    UNIQUE ("FK_note_id", "feature_name", "FK_strategy_id");

ALTER TABLE "{{prefix}}note_features" 
    ADD CONSTRAINT "{{prefix}}note_features_FK_note_id_fk" 
    FOREIGN KEY ("FK_note_id")
    REFERENCES "{{prefix}}notes"("id")
    ON DELETE RESTRICT
    ON UPDATE RESTRICT;
    
ALTER TABLE "{{prefix}}note_features" 
    ADD CONSTRAINT "{{prefix}}note_features_FK_note_section_id_fk" 
    FOREIGN KEY ("FK_note_section_id")
    REFERENCES "{{prefix}}note_sections"("id")
    ON DELETE RESTRICT
    ON UPDATE RESTRICT;

ALTER TABLE "{{prefix}}note_features" 
    ADD CONSTRAINT "{{prefix}}note_features_FK_strategy_id_fk" 
    FOREIGN KEY ("FK_strategy_id")
    REFERENCES "{{prefix}}strategies"("id")
    ON DELETE RESTRICT
    ON UPDATE RESTRICT;

CREATE INDEX IF NOT EXISTS "{{prefix}}note_features_FK_strategy_id_key" 
    ON "{{prefix}}note_features" ("FK_strategy_id");

CREATE TABLE "{{prefix}}annotators" (
    "id"            BIGSERIAL PRIMARY KEY,
    "username"      VARCHAR(255) UNIQUE NOT NULL,
    "password"      TEXT,
    "admin"         BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE TABLE "{{prefix}}episode_labels" (
    "id"                    BIGSERIAL PRIMARY KEY,
    "FK_note_feature_id"    BIGINT,
    "FK_episode_id"         BIGINT NOT NULL,
    "FK_strategy_id"        BIGINT,
    "FK_task_id"            BIGINT NOT NULL,
    "task_name"             VARCHAR(255) NOT NULL,
    "label_name"            TEXT NOT NULL,
    "label_value"           NUMERIC(10, 8),
    "line_number"           BIGINT,
    "FK_human_annotator"    VARCHAR(255)
);

ALTER TABLE "{{prefix}}episode_labels" 
    ADD CONSTRAINT "{{prefix}}episode_labels_episode_feat_strategy_label_key" 
    UNIQUE ("FK_episode_id", "FK_task_id", "FK_strategy_id", "FK_human_annotator", 
        "label_name");

ALTER TABLE "{{prefix}}episode_labels"
    ADD CONSTRAINT "{{prefix}}episode_labels_either_strategy_or_human"
    CHECK ("FK_strategy_id" IS NOT NULL OR "FK_human_annotator" IS NOT NULL);

ALTER TABLE "{{prefix}}episode_labels" 
    ADD CONSTRAINT "{{prefix}}episode_labels_FK_note_feature_id_fk" 
    FOREIGN KEY ("FK_note_feature_id")
    REFERENCES "{{prefix}}note_features"("id")
    ON DELETE RESTRICT
    ON UPDATE RESTRICT;

ALTER TABLE "{{prefix}}episode_labels" 
    ADD CONSTRAINT "{{prefix}}episode_labels_FK_episode_id_fk" 
    FOREIGN KEY ("FK_episode_id")
    REFERENCES "{{episode_table}}"("id")
    ON DELETE RESTRICT
    ON UPDATE RESTRICT;

ALTER TABLE "{{prefix}}episode_labels" 
    ADD CONSTRAINT "{{prefix}}episode_labels_FK_strategy_id_fk" 
    FOREIGN KEY ("FK_strategy_id")
    REFERENCES "{{prefix}}strategies"("id")
    ON DELETE RESTRICT
    ON UPDATE RESTRICT;

ALTER TABLE "{{prefix}}episode_labels" 
    ADD CONSTRAINT "{{prefix}}episode_labels_FK_task_id_fk" 
    FOREIGN KEY ("FK_task_id")
    REFERENCES "{{prefix}}tasks"("id")
    ON DELETE RESTRICT
    ON UPDATE RESTRICT;

ALTER TABLE "{{prefix}}episode_labels" 
    ADD CONSTRAINT "{{prefix}}episode_labels_FK_human_annotator_fk" 
    FOREIGN KEY ("FK_human_annotator")
    REFERENCES "{{prefix}}annotators"("username")
    ON DELETE RESTRICT
    ON UPDATE RESTRICT;