import React, { createContext, useContext, useState } from 'react';
import { UploadResponse, CohortPredictionResponse, StudentPrediction } from '../types/api';
import { PageId } from '../components/layout/Sidebar';

export interface CohortContextType {
  file: File | null;
  setFile: (file: File | null) => void;
  fileName: string | null;
  setFileName: (name: string | null) => void;
  uploadResult: UploadResponse | null;
  setUploadResult: (res: UploadResponse | null) => void;
  mappings: Record<string, string>;
  setMappings: (mappings: Record<string, string>) => void;
  predictionResult: CohortPredictionResponse | null;
  setPredictionResult: (res: CohortPredictionResponse | null) => void;
  selectedStudent: StudentPrediction | null;
  setSelectedStudent: (stu: StudentPrediction | null) => void;
  currentPage: PageId;
  setCurrentPage: (page: PageId) => void;
  resetCohort: () => void;
}

export const STORAGE_KEY_UPLOAD = 'pri_cohort_upload';
export const STORAGE_KEY_PRED = 'pri_cohort_prediction';
export const STORAGE_KEY_MAPPINGS = 'pri_cohort_mappings';
export const STORAGE_KEY_FILENAME = 'pri_cohort_filename';

export const CohortContext = createContext<CohortContextType | undefined>(undefined);
