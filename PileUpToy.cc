#include <TCanvas.h>
#include <TF1.h>
#include <TFile.h>
#include <TGraph.h>
#include <TH1D.h>
#include <TLegend.h>
#include <TMath.h>
#include <TRandom3.h>
#include <TStyle.h>

#include <vector>
#include <cmath>

constexpr double kITSreadoutRate{0.2e6};
constexpr double kMinSeparation{.1};
constexpr double kBunchCrossingRate{2.e7};
constexpr double kMu{0.05};
constexpr long kNumberOfROframes{4000000};
constexpr int kTrigger{150};
constexpr int kFitRejectionLimit{0};

struct vertex {
  std::vector<int> ids;
  double res;
  double z;
  int n;
  int bc;
  bool merged;
  bool inBunchPU;
};

void PileUpToy(const double itsRate, const int fitRejection) {
  gRandom->SetSeed(123456);

  TFile multFile("mult.root");
  TH1* multHist = (TH1*)multFile.Get("multHist");

  TF1* pvResolution = new TF1("pvResolution","([0]+[1]/sqrt(x)) * 1.e-4",0.1,1010);
  pvResolution->SetParameters(-0.157435, 99.3137);

  TH1D* hNvert = new TH1D("hNvert",";Number of vertices per RO frame;Counts",21,-0.5,20.5);
  TH1D* hNvertBC = new TH1D("hNvertBC",";Number of vertices per BC;Counts",21,-0.5,20.5);
  TH1D* hMult = new TH1D("hMult","Total multiplicity;Generated multiplicity;Events",300,0.5,300.5);
  TH1D* hSelectedMult = new TH1D("hSelectedMult","Event processed by the ITS reco;Generated multiplicity;Events",300,0.5,300.5);
  TH1D* hSelectedGood = new TH1D("hSelectedGood","Good triggers;Generated multiplicity;Events",300,0.5,300.5);
  TH1D* hSelectedFake = new TH1D("hSelectedFake","Pile-up of 2 vertices;Generated multiplicity;Events",300,0.5,300.5);
  TH1D* hSelectedFake3 = new TH1D("hSelectedFake3","Pile-up of 3 or more vertices;Generated multiplicity;Events",300,0.5,300.5);
  TH1D* hSelectedFakeAll = new TH1D("hSelectedFakeAll","Fake triggers (all vertices);Generated multiplicity;Events",300,0.5,300.5);
  TH1D* hSelectedFakeOOB = new TH1D("hSelectedFakeOOB","Fake triggers (OOB);Generated multiplicity;Events",300,0.5,300.5);
  TH1D* hSelectedHalf = new TH1D("hSelectedHalf","Fake triggers;Generated multiplicity;Events",300,0.5,300.5);
  hMult->SetLineColor(kBlack);
  hSelectedMult->SetLineColor(kBlack);
  hSelectedFakeOOB->SetLineColor(kRed);
  hSelectedFakeOOB->SetFillColor(kRed);
  hSelectedFakeOOB->SetFillStyle(3345);
  hSelectedFakeAll->SetLineColor(kRed);
  hSelectedFakeAll->SetFillColor(kRed);
  hSelectedFakeAll->SetFillStyle(3345);
  hSelectedFake->SetLineColor(kRed);
  hSelectedFake->SetFillColor(kRed);
  hSelectedFake->SetFillStyle(3345);
  hSelectedFake3->SetLineColor(kRed+2);
  hSelectedFake3->SetFillColor(kRed+2);
  hSelectedFake3->SetFillStyle(3356);
  hSelectedGood->SetLineColor(kGreen+3);
  hSelectedGood->SetFillColor(kGreen+3);
  hSelectedGood->SetFillStyle(3354);
  TH1D* hDeltaZ = new TH1D("hDeltaZ",";#Deltaz (cm);Events",400,0,20);
  TH1D* hSigmaZ = new TH1D("hSigmaZ",";#sigma_{z} (cm);Events",400,0,0.5);
  TH1D* hSigmaZall = new TH1D("hSigmaZall",";#sigma_{z} (cm);Events",400,0,0.5);
  
  std::vector<vertex> genVertices;
  std::vector<vertex> recVertices;

  const int nBCperRO = kBunchCrossingRate / itsRate;

  int triggeredROframe{0};
  int physicsTriggeredROframe{0};
  int fitSelected{0};
  for (int iROframe{0}; iROframe < kNumberOfROframes; ++iROframe) {
    genVertices.clear();
    recVertices.clear();
    int nVert{0};
    bool interestingRO{false};
    bool fitVETO{true};
    for (int iBC{0}; iBC < nBCperRO; ++iBC) {
      int nVertBC = gRandom->Poisson(kMu);
      for (int iVert{0}; iVert < nVertBC; ++iVert) {
        int mult = int(multHist->GetRandom());
        if (!mult) mult = 1;
        vertex vtx;
        vtx.ids.push_back(genVertices.size());
        vtx.res = pvResolution->Eval(mult);
        vtx.z = gRandom->Gaus(0,5);
        vtx.n = mult;
        vtx.merged = false;
        vtx.bc = iBC;
        vtx.inBunchPU = false;
        genVertices.emplace_back(vtx);
        hMult->Fill(mult);
        if (vtx.n > kTrigger && !interestingRO) {
          physicsTriggeredROframe++;
          interestingRO = true;
        }
        if (mult > fitRejection)
          fitVETO = false;
      }
      nVert+=nVertBC;
      hNvertBC->Fill(nVertBC);
    }
    hNvert->Fill(nVert);

    if (fitVETO)
      continue;
    else
      fitSelected++;

    for (const auto& vert : genVertices)
      hSelectedMult->Fill(vert.n);

    recVertices = genVertices;

    bool merging{false};
    bool first{true};
    do {
      merging = false;
      for (size_t iVert{0}; iVert < recVertices.size() && !merging; ++iVert) {
        for (size_t jVert{iVert + 1}; jVert < recVertices.size(); ++jVert) {
          double sigma{std::hypot(recVertices[iVert].res,recVertices[jVert].res)};
          double delta{std::abs(recVertices[iVert].z - recVertices[jVert].z)};
          if (first) {
            hDeltaZ->Fill(delta);
            hSigmaZ->Fill(10 * sigma);
          }
          hSigmaZall->Fill(10 * sigma);
          if (delta < kMinSeparation) {
              merging = true;
              recVertices[iVert].merged = true;
              recVertices[iVert].z = (recVertices[iVert].z * recVertices[iVert].n + recVertices[jVert].z * recVertices[jVert].n) / (recVertices[iVert].n + recVertices[jVert].n);
              recVertices[iVert].n = recVertices[iVert].n + recVertices[jVert].n;
              recVertices[iVert].res = pvResolution->Eval(recVertices[iVert].n);
              recVertices[iVert].inBunchPU = (recVertices[iVert].bc == recVertices[jVert].bc);
              recVertices[iVert].ids.insert(recVertices[iVert].ids.end(), recVertices[jVert].ids.begin(), recVertices[jVert].ids.end());
              recVertices.erase(recVertices.begin()+jVert);
              break;
          }
        }
      }
      first = false;
    } while (merging);

    bool triggered{false};
    for (auto& vert : recVertices) {
      if (vert.n > kTrigger) {
        triggered = true;
        if (vert.merged) {
          bool half{false};
          int max {0};
          int sum {0};
          for (int id : vert.ids) {
            if (genVertices[id].n > kTrigger)
              half = true;
            if (genVertices[id].n > max)
              max = genVertices[id].n;
            sum += genVertices[id].n;
          }
          if (vert.ids.size() == 2)
            hSelectedFake->Fill(max);
          else
            hSelectedFake3->Fill(max);
          if (!vert.inBunchPU)
            hSelectedFakeOOB->Fill(max);
          for (int id : vert.ids) {
            hSelectedFakeAll->Fill(genVertices[id].n);
            if (half)
              hSelectedHalf->Fill(genVertices[id].n);
          }
        } else
          hSelectedGood->Fill(vert.n);
      }
    }
    if (triggered)
      triggeredROframe++;
  }

  double fitRate = fitSelected / double(kNumberOfROframes) * itsRate;
  std::string fitString = fitRejection > 0 ? Form("FIT selects N_{ch} > %i (%.1f kHz)",fitRejection, fitRate * 1.e-3) : "No FIT rejection";
  double triggerRate = triggeredROframe / double(kNumberOfROframes) * itsRate;
  double physicsRate = physicsTriggeredROframe / double(kNumberOfROframes) * itsRate;

  TFile output("output.root","recreate");
  TCanvas finalCv("finalCv");
  finalCv.cd();
  finalCv.SetLogy();
  hSelectedMult->Draw();
  hSelectedGood->Draw("same");
  hSelectedFake->Draw("same");
  hSelectedFake3->Draw("same");
  TLegend* leg = finalCv.BuildLegend(0.5,0.5,0.9,0.82,Form("#splitline{#splitline{ITS RO rate: %1.1f MHz}{%s}}{Trigger/Process rates: %.0f/%.0f Hz}",itsRate * 1.e-6,fitString.data(),triggerRate,physicsRate));
  leg->SetMargin(0.15);
  leg->SetFillStyle(0);
  finalCv.SaveAs(Form("trigger_%.0f_%i.pdf",itsRate*1.e-3,fitRejection));
  finalCv.Write();
  hNvert->Write();
  hNvertBC->Write();
  hMult->Write();
  hSelectedFake->Write();
  hSelectedFake3->Write();
  hSelectedGood->Write();
  hSelectedFakeAll->Write();
  hSelectedFakeOOB->Write();
  hSelectedHalf->Write();
  hDeltaZ->Write();
  hSigmaZ->Write();
  hSigmaZall->Write();
  output.Close();

}

void PileUpToy() {
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);

  PileUpToy(kITSreadoutRate,0);
  PileUpToy(kITSreadoutRate * 2,0);
  // PileUpToy(kITSreadoutRate * 5,0);
  PileUpToy(kITSreadoutRate,50);
  PileUpToy(kITSreadoutRate * 2,50);
  // PileUpToy(kITSreadoutRate * 5,50);
}