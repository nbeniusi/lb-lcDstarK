import ROOT
df = ROOT.RDataFrame(10).Define("x","rdfentry_")
h = df.Histo1D(("h","test",10,0,10),"x")
c = ROOT.TCanvas(); h.Draw(); c.SaveAs("figures/rdf_test.png")
