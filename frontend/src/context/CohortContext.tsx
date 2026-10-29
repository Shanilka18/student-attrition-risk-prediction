import React, { createContext, useContext, useState, useEffect } from 'react';
import { UploadResponse, CohortPredictionResponse, StudentPrediction } from '../types/api';
import { PageId } from '../components/layout/Sidebar';

interface CohortContextType {
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

const CohortContext = createContext<CohortContextType | undefined>(undefined);

const STORAGE_KEY_UPLOAD = 'pri_cohort_upload';
const STORAGE_KEY_PRED = 'pri_cohort_prediction';
const STORAGE_KEY_MAPPINGS = 'pri_cohort_mappings';
const STORAGE_KEY_FILENAME = 'pri_cohort_filename';

export const CohortProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [currentPage, setCurrentPage] = useState<PageId>('overview');
  const [file, setFile] = useState<File | null>(null);
  const [fileName, setFileName] = useState<string | null>(() => {
    return sessionStorage.getItem(STORAGE_KEY_FILENAME);
  });

  const [uploadResult, setUploadResultState] = useState<UploadResponse | null>(() => {
    try {
      const saved = sessionStorage.getItem(STORAGE_KEY_UPLOAD);
      return saved ? JSON.parse(saved) : null;
    } catch {
      return null;
    }
  });

  const [mappings, setMappingsState] = useState<Record<string, string>>(() => {
    try {
      const saved = sessionStorage.getItem(STORAGE_KEY_MAPPINGS);
      return saved ? JSON.parse(saved) : {};
    } catch {
      return {};
    }
  });

  const [predictionResult, setPredictionResultState] = useState<CohortPredictionResponse | null>(() => {
    try {
      const saved = sessionStorage.getItem(STORAGE_KEY_PRED);
      return saved ? JSON.parse(saved) : null;
    } catch {
      return null;
    }
  });

  const [selectedStudent, setSelectedStudent] = useState<StudentPrediction | null>(() => {
    try {
      const saved = sessionStorage.getItem(STORAGE_KEY_PRED);
      if (saved) {
        const parsed: CohortPredictionResponse = JSON.parse(saved);
        return parsed.students && parsed.students.length > 0 ? parsed.students[0] : null;
      }
    } catch {
      // ignore
    }
    return null;
  });

  const setUploadResult = (res: UploadResponse | null) => {
    setUploadResultState(res);
    try {
      if (res) {
        sessionStorage.setItem(STORAGE_KEY_UPLOAD, JSON.stringify(res));
      } else {
        sessionStorage.removeItem(STORAGE_KEY_UPLOAD);
      }
    } catch (e) {
      console.warn('Could not persist uploadResult to sessionStorage:', e);
    }
  };

  const setMappings = (newMappings: Record<string, string>) => {
    setMappingsState(newMappings);
    try {
      sessionStorage.setItem(STORAGE_KEY_MAPPINGS, JSON.stringify(newMappings));
    } catch (e) {
      console.warn('Could not persist mappings to sessionStorage:', e);
    }
  };

  const setPredictionResult = (res: CohortPredictionResponse | null) => {
    setPredictionResultState(res);
    try {
      if (res) {
        sessionStorage.setItem(STORAGE_KEY_PRED, JSON.stringify(res));
        if (res.students && res.students.length > 0 && !selectedStudent) {
          setSelectedStudent(res.students[0]);
        }
      } else {
        sessionStorage.removeItem(STORAGE_KEY_PRED);
      }
    } catch (e) {
      console.warn('Could not persist predictionResult to sessionStorage (exceeded quota?):', e);
    }
  };

  const handleSetFileName = (name: string | null) => {
    setFileName(name);
    try {
      if (name) {
        sessionStorage.setItem(STORAGE_KEY_FILENAME, name);
      } else {
        sessionStorage.removeItem(STORAGE_KEY_FILENAME);
      }
    } catch (e) {
      console.warn('Could not persist filename:', e);
    }
  };

  const resetCohort = () => {
    setFile(null);
    handleSetFileName(null);
    setUploadResult(null);
    setMappings({});
    setPredictionResult(null);
    setSelectedStudent(null);
    try {
      sessionStorage.removeItem(STORAGE_KEY_UPLOAD);
      sessionStorage.removeItem(STORAGE_KEY_PRED);
      sessionStorage.removeItem(STORAGE_KEY_MAPPINGS);
      sessionStorage.removeItem(STORAGE_KEY_FILENAME);
    } catch (e) {
      console.warn('Could not clear sessionStorage:', e);
    }
  };

  return (
    <CohortContext.Provider
      value={{
        file,
        setFile,
        fileName,
        setFileName: handleSetFileName,
        uploadResult,
        setUploadResult,
        mappings,
        setMappings,
        predictionResult,
        setPredictionResult,
        selectedStudent,
        setSelectedStudent,
        currentPage,
        setCurrentPage,
        resetCohort,
      }}
    >
      {children}
    </CohortContext.Provider>
  );
};

export const useCohort = (): CohortContextType => {
  const context = useContext(CohortContext);
  if (!context) {
    throw new Error('useCohort must be used within a CohortProvider');
  }
  return context;
};
