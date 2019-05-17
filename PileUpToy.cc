#include <TCanvas.h>
#include <TF1.h>
#include <TFile.h>
#include <TGraph.h>
#include <TH1D.h>
#include <TLatex.h>
#include <TLegend.h>
#include <TMath.h>
#include <TRandom3.h>
#include <TStyle.h>

#include <vector>
#include <cmath>
#include <TMath.h>

namespace {
constexpr double kITSreadoutRate{0.2e6};
constexpr double kNsigmaSeparation{5};
constexpr double kBunchCrossingRate{2.e7};
constexpr double kMu{0.05};
constexpr long kNumberOfROframes{4000000};
constexpr int kTrigger{150};
constexpr int kFitRejectionLimit{0};
constexpr double kTrackletVertexResolution{0.01}; //100um
const double kRMScut = kTrackletVertexResolution * (1. + 5. / std::sqrt(kTrigger));

struct vertex {
  std::vector<int> ids;
  std::vector<double> tracklets;
  double res;
  double z;
  int n;
  int bc;
  bool merged;
  bool inBunchPU;
};
}

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
  TH1D* hRMSsingle = new TH1D("hRMSsingle","Single vertex;RMS (#mum);Counts",600,kTrackletVertexResolution * 7000, 100000 * kTrackletVertexResolution);
  TH1D* hRMSpileup = new TH1D("hRMSpileup","Pile-up;RMS (#mum);RMS (#mum);Counts",600,kTrackletVertexResolution * 7000, 100000 * kTrackletVertexResolution);
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
  hRMSsingle->SetLineColor(kGreen+3);
  hRMSsingle->SetFillColor(kGreen+3);
  hRMSsingle->SetFillStyle(3354);
  hRMSpileup->SetLineColor(kRed);
  hRMSpileup->SetFillColor(kRed);
  hRMSpileup->SetFillStyle(3345);
  TH1D* hDeltaZ = new TH1D("hDeltaZ",";#Deltaz (cm);Events",400,0,20);
  TH1D* hSigmaZ = new TH1D("hSigmaZ",";#sigma_{z} (cm);Events",400,0,0.5);
  TH1D* hSigmaZall = new TH1D("hSigmaZall",";#sigma_{z} (cm);Events",400,0,0.5);
  
  std::vector<vertex> genVertices;
  std::vector<vertex> recVertices;

  const int nBCperRO = kBunchCrossingRate / itsRate;

  int triggeredROframe{0};
  int physicsTriggeredROframe{0};
  int fitSelected{0};
  int spuriousSelections{0};
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
        vtx.z = gRandom->Gaus(0,5); /// initialise the position
        for (int iTrkl{0}; iTrkl < mult; ++iTrkl)
          vtx.tracklets.push_back(gRandom->Gaus(vtx.z, kTrackletVertexResolution));
        vtx.z = TMath::Mean(vtx.tracklets.begin(), vtx.tracklets.end()); /// compute the actual average of the tracklet z intercepts
        vtx.ids.push_back(genVertices.size());
        vtx.res = TMath::RMS(vtx.tracklets.begin(), vtx.tracklets.end());
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
            hSigmaZ->Fill(kNsigmaSeparation * sigma);
          }
          hSigmaZall->Fill(kNsigmaSeparation * sigma);
          if (delta < sigma * kNsigmaSeparation) {
              merging = true;
              recVertices[iVert].merged = true;
              recVertices[iVert].tracklets.insert(recVertices[iVert].tracklets.begin(), recVertices[jVert].tracklets.begin(), recVertices[jVert].tracklets.end());
              recVertices[iVert].res = TMath::RMS(recVertices[iVert].tracklets.begin(), recVertices[iVert].tracklets.end());
              recVertices[iVert].z = TMath::Mean(recVertices[iVert].tracklets.begin(), recVertices[iVert].tracklets.end());
              recVertices[iVert].n = recVertices[iVert].tracklets.size();
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
    bool badTrigger{false};
    for (auto& vert : recVertices) {
      if (vert.n > kTrigger && vert.res < kRMScut) {
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
          hRMSpileup->Fill(vert.res * 1.e4);
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
          if (!half)
            badTrigger = true;
        } else {
          hRMSsingle->Fill(vert.res * 1.e4);
          hSelectedGood->Fill(vert.n);
          badTrigger = false;
        }
      }
    }
    if (triggered)
      triggeredROframe++;
    if (badTrigger)
      spuriousSelections++;
  }

  double triggerEfficiency = (triggeredROframe - spuriousSelections) / double(physicsTriggeredROframe);
  double triggerContamination = (spuriousSelections) / double(triggeredROframe);
  double fitRate = fitSelected / double(kNumberOfROframes) * itsRate;
  double triggerRate = triggeredROframe / double(kNumberOfROframes) * itsRate;
  double physicsRate = physicsTriggeredROframe / double(kNumberOfROframes) * itsRate;

  TLatex tex;
  tex.SetTextFont(42);
  tex.SetTextSize(0.04);
  tex.SetTextSize(0.04);
  tex.SetTextAlign(33);
  std::string fitString = fitRejection > 0 ? Form("FIT selects N_{ch} > %i (%.1f kHz)",fitRejection, fitRate * 1.e-3) : "No FIT rejection";
  std::string headerText = Form("ITS RO rate: %.0f kHz, %s",itsRate * 1.e-3,fitString.data());
  std::string legendText = Form("#splitline{Trigger/Process rates: %.0f/%.0f Hz}{Trigger efficiency/purity: %.1f%%/%.1f%%}",triggerRate, physicsRate,triggerEfficiency * 100, (1 - triggerContamination)*100);

  TFile output("output.root","recreate");
  TCanvas finalCv("finalCv");
  finalCv.cd();
  finalCv.SetLogy();
  hSelectedMult->Draw();
  hSelectedGood->Draw("same");
  hSelectedFake->Draw("same");
  hSelectedFake3->Draw("same");
  TLegend* leg = finalCv.BuildLegend(0.5,0.5,0.9,0.82,legendText.data());
  leg->SetMargin(0.15);
  leg->SetFillStyle(0);
  tex.DrawLatexNDC(0.95,0.95,headerText.data());
  finalCv.SaveAs(Form("trigger_%.0f_%i.pdf",itsRate*1.e-3,fitRejection));
  finalCv.Write();
  TCanvas rmsCv("rmsCv");
  rmsCv.SetLogy();
  if (hRMSsingle->GetMaximum() < hRMSpileup->GetMaximum())
    hRMSsingle->SetMaximum(hRMSpileup->GetMaximum());
  hRMSsingle->Draw();
  hRMSpileup->Draw("same");
  rmsCv.BuildLegend(0.5,0.6,0.9,0.85,legendText.data());
  tex.DrawLatexNDC(0.95,0.95,headerText.data());
  rmsCv.SaveAs(Form("rms_%.0f_%i.pdf",itsRate*1.e-3,fitRejection));
  rmsCv.Write();
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
  hRMSsingle->Write();
  hRMSpileup->Write();
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