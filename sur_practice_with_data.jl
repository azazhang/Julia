using LinearAlgebra, Distributions

# #Set up
# #need to calculate VCV matrix per Hamilton:
# #omegahat_0 = (1/T)SUM[y_t - X'_t*betahat_0][y_t - X'_t*betahat_0]'
# #or... can we use the residuals from OLS regressions?
# corr_true = -.45
# Sigma_true = [.5 corr_true*sqrt(.5)*sqrt(.3); corr_true*sqrt(.5)*sqrt(.3) .3]
# H_true = LinearAlgebra.cholesky(Sigma_true).U

#Generate data
function embed(x,p)
    n = length(x)
    pp = p+1
    m = zeros(n-pp+1,pp)
    for i in 1:pp
        m[:,i] = x[(pp-i+1):(n-i+1)]
    end
    return m
end

using CSV, DataFrames, Distributions, LinearAlgebra

#Read Time Series Data
df1 = DataFrame(CSV.read("JuliaTSvars.csv"; header=["DATE", "logdiffEPRu", "logdiffUER", "logdiffRMW", "logdiffRAHE"], datarow=14))

EPR = df1[:,2]
UER = df1[:,3]
MW = df1[:,4]
AHE = df1[:,5]

#generate lags of variables
EPRlags1 = embed(EPR, 13)
UERlags1 = embed(UER,13)
MWlags1 = embed(MW,13)
AHElags1 = embed(AHE,13)

#drop contemporaneous value
#if 4 eqns, we may need 4 X matrices
EPRlags = convert(Matrix, EPRlags1[:,2:14])
UERlags = convert(Matrix, UERlags1[:,2:14])
MWlags = convert(Matrix, MWlags1[:,2:14])
AHElags = convert(Matrix, AHElags1[:,2:14])

# Xepr = [EPRlags1[:,2:14] UERlags1 AHElags1 MWlags1]
# Xuer = [EPRlags1 UERlags1[:,2:14] AHElags1 MWlags1]
# Xahe = [EPRlags1 UERlags1 AHElags1[:,2:14] MWlags1]
# Xmw = [EPRlags1 UERlags1 AHElags1 MWlags1[:,2:14]]

XSUR = [EPRlags UERlags AHElags MWlags]

yepr = EPRlags1[:,1]
yuer = UERlags1[:,1]
yahe = AHElags1[:,1]
ymw = MWlags1[:,1]

y1 = yepr
y2 = yuer
y3 = yahe
y4 = ymw

ySUR = [EPRlags1[:,1];UERlags1[:,1];AHElags1[:,1];MWlags1[:,1]]

nobs = size(y1)[1]

# #Data Generating stuff
# for i in 1:nobs
#     error_vec = H_true'*(randn(2,1))
#     x1[i] = rand(Normal(1,1))
#     x2[i] = rand(Normal(1,1))
#     y1[i] = beta11 + beta12*x1[i] + error_vec[1]
#     y2[i] = beta21 + beta22*x2[i] + error_vec[2]
#
#     return x1, x2, y1, y2
# end


#Define some terms that will be useful later when running the Gibbs
#sampler.
#bigy = ySUR #not used
bigX1 = [ones(size(XSUR)[1]) XSUR]
# bigX2 = bigX1 #since all X's are the same
# bigX3 = bigX1
# bigX4 = bigX1
X1X1 = bigX1'*bigX1
#no need to repeat these since all X's are the same
# X1X2 = bigX1'*bigX2
# X1X3 = bigX1'*bigX3
# X1X4 =
# X2X1 = bigX2'*bigX1
# X2X2 = bigX2'*bigX2

#recall, that later on, it may call for X2Y1 (for example) and all X's are the same
X1Y1 = bigX1'*y1
X1Y2 = bigX1'*y2
X1Y3 = bigX1'*y3
X1Y4 = bigX1'*y4
ytilde = zeros(4,nobs)
xtilde = zeros(4,212,nobs)
#ytilde: #eqns & nobs, xtilde:#eqns, total variables, nobs
for i in 1:nobs
    ytilde[:,i] = [y1[i,1] y2[i,1] y3[i,1] y4[i,1]]'
    xtilde[:,:,i] = [1 XSUR[i,:]' zeros(159)';
                    zeros(53)' 1 XSUR[i,:]' zeros(106)';
                    zeros(106)' 1 XSUR[i,:]' zeros(53)';
                    zeros(159)' 1 XSUR[i,:]']

#    return ytilde, xtilde
end

#Define the prior hyperparameters;
β0 = zeros(212,1)
#change B0 and β0 according to the number of (total) variables
B0 = zeros(212,212)+I*1000
Omega = zeros(4,4)+I #R0 in Greenberg
v0 = 0 #what are appropriate values for this?
v1 = v0 + 53 #v0 + number of parameters

#Gibbs sampler
iter = 1000
burn = 200
sig1_final = zeros(iter-burn,1)
sig2_final = zeros(iter-burn,1)
sig3_final = zeros(iter-burn,1)
sig4_final = zeros(iter-burn,1)
corr12_final = zeros(iter-burn,1)
corr13_final = zeros(iter-burn,1)
corr14_final = zeros(iter-burn,1)
corr23_final = zeros(iter-burn,1)
corr24_final = zeros(iter-burn,1)
corr34_final = zeros(iter-burn,1)
betas_final = zeros(iter-burn,212)
B1_final = zeros(212,212)
#be sure to change betas_final to reflect no. of total variables

Sig1 = 1
Sig2 = 1
Sig3 = 1
Sig4 = 1
Sig12 = 0
Sig13 = 0
Sig14 = 0
Sig23 = 0
Sig24 = 0
Sig34 = 0

function sur_gibbs_ian(Sig1, Sig2, Sig3, Sig4, Sig12, Sig13, Sig14, Sig23, Sig24, Sig34, B0, β0, Omega, v0)
    for i in 1:iter
    #for i in 1:5 #test loop
        #Sample from Conditional for Beta
        #note X1X1 stands in for X1X2, X1X3, etc since all X's are the same
        matrix1 = [Sig1*X1X1 Sig12*X1X1 Sig13*X1X1 Sig14*X1X1;
                   Sig12*X1X1 Sig2*X1X1 Sig23*X1X1 Sig24*X1X1;
                   Sig13*X1X1 Sig23*X1X1 Sig3*X1X1 Sig34*X1X1;
                   Sig14*X1X1 Sig24*X1X1 Sig34*X1X1 Sig4*X1X1]
        matrix2 = [ Sig1*X1Y1 + Sig12*X1Y2 + Sig13*X1Y3 + Sig14*X1Y4;
                    Sig12*X1Y1 + Sig2*X1Y2 + Sig23*X1Y3 + Sig24*X1Y4;
                    Sig13*X1Y1 + Sig23*X1Y2 + Sig3*X1Y3 + Sig34*X1Y4;
                    Sig14*X1Y1 + Sig24*X1Y2 + Sig34*X1Y3 + Sig4*X1Y4]
        B1 = inv(matrix1 + inv(B0))
        β1 = matrix2 + inv(B0)*β0
        #carryover from code I found... what does this do?
        H_beta = LinearAlgebra.cholesky(Symmetric(B1)).U
        betas = B1*β1 + H_beta'*randn(212,1)
            #randn(212,1) = total number of variables aka, one row of Xj

        #Sample from Wishart conditional for Sigmainv
        tempp = zeros(4,4)
        for j in 1:nobs
            temp_term = (ytilde[:,j] - xtilde[:,:,j]*betas)*(ytilde[:,j] - xtilde[:,:,j]*betas)'
            tempp = temp_term + tempp
        end

        #inv(tempp + inv(Omega)) is R1 in Greenberg
        R1 = inv(tempp + inv(Omega))
        R1Sym = LinearAlgebra.cholesky(Hermitian(R1)).U'*LinearAlgebra.cholesky(Hermitian(R1)).U
        Sigmainv = rand(Wishart(nobs+v1, R1Sym))
        Sigma = inv(Sigmainv)
        Sig1 = Sigmainv[1,1]
        Sig2 = Sigmainv[2,2]
        Sig3 = Sigmainv[3,3]
        Sig4 = Sigmainv[4,4]
        Sig12 = Sigmainv[1,2]
        Sig13 = Sigmainv[1,3]
        Sig14 = Sigmainv[1,4]
        Sig23 = Sigmainv[2,3]
        Sig24 = Sigmainv[2,4]
        Sig34 = Sigmainv[3,4]

        s1 = Sigma[1,1]
        s2 = Sigma[2,2]
        s3 = Sigma[3,3]
        s4 = Sigma[4,4]
        s12 = Sigma[1,2]
        s13 = Sigma[1,3]
        s14 = Sigma[1,4]
        s23 = Sigma[2,3]
        s24 = Sigma[2,4]
        s34 = Sigma[3,4]

        #if i < burn #for testing
         if i > burn
             sig1_final[i-burn,1] = s1
             sig2_final[i-burn,1] = s2
             sig3_final[i-burn,1] = s3
             sig4_final[i-burn,1] = s4
             corr12_final[i-burn,1] = s12/(sqrt(s1)*sqrt(s2))
             corr13_final[i-burn,1] = s13/(sqrt(s1)*sqrt(s3))
             corr14_final[i-burn,1] = s14/(sqrt(s1)*sqrt(s4))
             corr23_final[i-burn,1] = s23/(sqrt(s2)*sqrt(s3))
             corr24_final[i-burn,1] = s24/(sqrt(s2)*sqrt(s4))
             corr34_final[i-burn,1] = s34/(sqrt(s3)*sqrt(s4))
             betas_final[i-burn,:] = betas'
             B1_final[:,:] = B1
         end
    end

    return sig1_final, sig2_final, sig3_final, sig4_final, corr12_final, corr13_final, corr14_final, corr23_final, corr24_final, corr34_final, betas_final, B1_final
end

output = sur_gibbs_ian(Sig1, Sig2, Sig3, Sig4,
    Sig12, Sig13, Sig14, Sig23, Sig24, Sig34,
    B0, β0, Omega, v0)


TSsig1 = mean(output[1])
TSsig2 = mean(output[2])
TSsig3 = mean(output[3])
TSsig4 = mean(output[4])
TSsig12 = mean(output[5])
TSsig13 = mean(output[6])
TSsig14 = mean(output[7])
TSsig23 = mean(output[8])
TSsig24 = mean(output[9])
TSsig34 = mean(output[10])
TSbeta = output[11]
TSB1 = output[12]

TSβ1 = zeros(size(TSbeta)[2],1)
for i in 1:length(TSβ1)
    TSβ1[i] = mean(TSbeta[:,i])
    return TSβ1
end

TSR1 = [TSsig1 TSsig12 TSsig13 TSsig14;
        TSsig12 TSsig2 TSsig23 TSsig24;
        TSsig13 TSsig23 TSsig3 TSsig34;
        TSsig14 TSsig24 TSsig34 TSsig4]


####PANEL DATA ESTIMATION

df2 = DataFrame(CSV.read("FinalPanelLong.csv"; header=["Date",	"EPRAL",	"EPRAK",	"EPRAZ",	"EPRAR",	"EPRCA",	"EPRCO",	"EPRCT",	"EPRDE",	"EPRDC",	"EPRFL", "EPRGA",	"EPRHI",	"EPRID",	"EPRIL",	"EPRIN",	"EPRIA",
    "EPRKS",	"EPRKY",	"EPRLA",	"EPRME",	"EPRMD",	"EPRMA",	"EPRMI",	"EPRMN",	"EPRMS",	"EPRMO",	"EPRMT",	"EPRNE",	"EPRNV",	"EPRNH",	"EPRNJ",	"EPRNM",	"EPRNY",	"EPRNC",	"EPRND",	"EPROH",
    "EPROK",	"EPROR",	"EPRPA",	"EPRRI",	"EPRSC",	"EPRSD",	"EPRTN",	"EPRTX",	"EPRUT",	"EPRVT",	"EPRVA",	"EPRWA",	"EPRWV",	"EPRWI",	"EPRWY",	"AHEAL",	"AHEAK",	"AHEAZ",	"AHEAR",	"AHECA",
    "AHECO",	"AHECT",	"AHEDE",	"AHEDC",	"AHEFL", "AHEGA",	"AHEHI",	"AHEID",	"AHEIL",	"AHEIN",	"AHEIA",	"AHEKS",	"AHEKY",	"AHELA",	"AHEME",	"AHEMD",	"AHEMA",	"AHEMI",	"AHEMN",	"AHEMS",
    "AHEMO",	"AHEMT",	"AHENE",	"AHENV",	"AHENH",	"AHENJ",	"AHENM",	"AHENY",	"AHENC",	"AHEND",	"AHEOH",	"AHEOK",	"AHEOR",	"AHEPA",	"AHERI",	"AHESC",	"AHESD",	"AHETN",	"AHETX",	"AHEUT",
    "AHEVT",	"AHEVA",	"AHEWA",	"AHEWV",	"AHEWI",	"AHEWY",	"EffMinWageAL",	"EffMinWageAK",	"EffMinWageAZ",	"EffMinWageAR",	"EffMinWageCA",	"EffMinWageCO",	"EffMinWageCT",	"EffMinWageDE",	"EffMinWageDC",	"EffMinWageFL",	"EffMinWageGA",
    "EffMinWageHI",	"EffMinWageID",	"EffMinWageIL",	"EffMinWageIN",	"EffMinWageIA",	"EffMinWageKS",	"EffMinWageKY",	"EffMinWageLA",	"EffMinWageME",	"EffMinWageMD",	"EffMinWageMA",	"EffMinWageMI",	"EffMinWageMN",	"EffMinWageMS",	"EffMinWageMO",	"EffMinWageMT",
    "EffMinWageNE",	"EffMinWageNV",	"EffMinWageNH",	"EffMinWageNJ",	"EffMinWageNM",	"EffMinWageNY",	"EffMinWageNC",	"EffMinWageND",	"EffMinWageOH",	"EffMinWageOK",	"EffMinWageOR",	"EffMinWagePA",	"EffMinWageRI",	"EffMinWageSC",	"EffMinWageSD",	"EffMinWageTN",
    "EffMinWageTX",	"EffMinWageUT",	"EffMinWageVT",	"EffMinWageVA",	"EffMinWageWA",	"EffMinWageWV",	"EffMinWageWI",	"EffMinWageWY",	"UERAL",	"UERAK",	"UERAZ",	"UERAR",	"UERCA",	"UERCO",	"UERCT",	"UERDE",	"UERDC",	"UERFL",
    "UERGA",	"UERHI",	"UERID",	"UERIL",	"UERIN",	"UERIA",	"UERKS",	"UERKY",	"UERLA",	"UERME",	"UERMD",	"UERMA",	"UERMI",	"UERMN",	"UERMS",	"UERMO",	"UERMT",	"UERNE",	"UERNV",	"UERNH",
    "UERNJ",	"UERNM",	"UERNY",	"UERNC",	"UERND",	"UEROH",	"UEROK",	"UEROR",	"UERPA",	"UERRI",	"UERSC",	"UERSD",	"UERTN",	"UERTX",	"UERUT",	"UERVT",	"UERVA",	"UERWA",	"UERWV",	"UERWI",
    "UERWY"], datarow=2))

#create lags of each state's variables
EPRmat = convert(Matrix, df2[:,2:52])
eprALlags = embed(EPRmat[:,1],13)
eprAKlags = embed(EPRmat[:,2],13)
eprAZlags = embed(EPRmat[:,3],13)
eprARlags = embed(EPRmat[:,4],13)
eprCAlags = embed(EPRmat[:,5],13)
eprCOlags = embed(EPRmat[:,6],13)
eprCTlags = embed(EPRmat[:,7],13)
eprDElags = embed(EPRmat[:,8],13)
eprDClags = embed(EPRmat[:,9],13)
eprFLlags = embed(EPRmat[:,10],13)
eprGAlags = embed(EPRmat[:,11],13)
eprHIlags = embed(EPRmat[:,12],13)
eprIDlags = embed(EPRmat[:,13],13)
eprILlags = embed(EPRmat[:,14],13)
eprINlags = embed(EPRmat[:,15],13)
eprIAlags = embed(EPRmat[:,16],13)
eprKSlags = embed(EPRmat[:,17],13)
eprKYlags = embed(EPRmat[:,18],13)
eprLAlags = embed(EPRmat[:,19],13)
eprMElags = embed(EPRmat[:,20],13)
eprMDlags = embed(EPRmat[:,21],13)
eprMAlags = embed(EPRmat[:,22],13)
eprMIlags = embed(EPRmat[:,23],13)
eprMNlags = embed(EPRmat[:,24],13)
eprMSlags = embed(EPRmat[:,25],13)
eprMOlags = embed(EPRmat[:,26],13)
eprMTlags = embed(EPRmat[:,27],13)
eprNElags = embed(EPRmat[:,28],13)
eprNVlags = embed(EPRmat[:,29],13)
eprNHlags = embed(EPRmat[:,30],13)
eprNJlags = embed(EPRmat[:,31],13)
eprNMlags = embed(EPRmat[:,32],13)
eprNYlags = embed(EPRmat[:,33],13)
eprNClags = embed(EPRmat[:,34],13)
eprNDlags = embed(EPRmat[:,35],13)
eprOHlags = embed(EPRmat[:,36],13)
eprOKlags = embed(EPRmat[:,37],13)
eprORlags = embed(EPRmat[:,38],13)
eprPAlags = embed(EPRmat[:,39],13)
eprRIlags = embed(EPRmat[:,40],13)
eprSClags = embed(EPRmat[:,41],13)
eprSDlags = embed(EPRmat[:,42],13)
eprTNlags = embed(EPRmat[:,43],13)
eprTXlags = embed(EPRmat[:,44],13)
eprUTlags = embed(EPRmat[:,45],13)
eprVTlags = embed(EPRmat[:,46],13)
eprVAlags = embed(EPRmat[:,47],13)
eprWAlags = embed(EPRmat[:,48],13)
eprWVlags = embed(EPRmat[:,49],13)
eprWIlags = embed(EPRmat[:,50],13)
eprWYlags = embed(EPRmat[:,51],13)

EPRlaglong = [eprALlags; 	eprAKlags; 	eprAZlags; 	eprARlags; 	eprCAlags;
    eprCOlags; 	eprCTlags; 	eprDElags; 	eprDClags; 	eprFLlags; 	eprGAlags;
    eprHIlags; 	eprIDlags; 	eprILlags; 	eprINlags; 	eprIAlags; 	eprKSlags;
    eprKYlags; 	eprLAlags; 	eprMElags; 	eprMDlags; 	eprMAlags; 	eprMIlags;
    eprMNlags; 	eprMSlags; 	eprMOlags; 	eprMTlags; 	eprNElags; 	eprNVlags;
    eprNHlags; 	eprNJlags; 	eprNMlags; 	eprNYlags; 	eprNClags; 	eprNDlags;
    eprOHlags; 	eprOKlags; 	eprORlags; 	eprPAlags; 	eprRIlags; 	eprSClags;
    eprSDlags; 	eprTNlags; 	eprTXlags; 	eprUTlags; 	eprVTlags; 	eprVAlags;
    eprWAlags; 	eprWVlags; 	eprWIlags; 	eprWYlags]

yEPRpanel = EPRlaglong[:,1]
xEPRpanel = EPRlaglong[:,2:end]

AHEmat = convert(Matrix, df2[:,53:103])
aheALlags = embed(AHEmat[:,1],13)
aheAKlags = embed(AHEmat[:,2],13)
aheAZlags = embed(AHEmat[:,3],13)
aheARlags = embed(AHEmat[:,4],13)
aheCAlags = embed(AHEmat[:,5],13)
aheCOlags = embed(AHEmat[:,6],13)
aheCTlags = embed(AHEmat[:,7],13)
aheDElags = embed(AHEmat[:,8],13)
aheDClags = embed(AHEmat[:,9],13)
aheFLlags = embed(AHEmat[:,10],13)
aheGAlags = embed(AHEmat[:,11],13)
aheHIlags = embed(AHEmat[:,12],13)
aheIDlags = embed(AHEmat[:,13],13)
aheILlags = embed(AHEmat[:,14],13)
aheINlags = embed(AHEmat[:,15],13)
aheIAlags = embed(AHEmat[:,16],13)
aheKSlags = embed(AHEmat[:,17],13)
aheKYlags = embed(AHEmat[:,18],13)
aheLAlags = embed(AHEmat[:,19],13)
aheMElags = embed(AHEmat[:,20],13)
aheMDlags = embed(AHEmat[:,21],13)
aheMAlags = embed(AHEmat[:,22],13)
aheMIlags = embed(AHEmat[:,23],13)
aheMNlags = embed(AHEmat[:,24],13)
aheMSlags = embed(AHEmat[:,25],13)
aheMOlags = embed(AHEmat[:,26],13)
aheMTlags = embed(AHEmat[:,27],13)
aheNElags = embed(AHEmat[:,28],13)
aheNVlags = embed(AHEmat[:,29],13)
aheNHlags = embed(AHEmat[:,30],13)
aheNJlags = embed(AHEmat[:,31],13)
aheNMlags = embed(AHEmat[:,32],13)
aheNYlags = embed(AHEmat[:,33],13)
aheNClags = embed(AHEmat[:,34],13)
aheNDlags = embed(AHEmat[:,35],13)
aheOHlags = embed(AHEmat[:,36],13)
aheOKlags = embed(AHEmat[:,37],13)
aheORlags = embed(AHEmat[:,38],13)
ahePAlags = embed(AHEmat[:,39],13)
aheRIlags = embed(AHEmat[:,40],13)
aheSClags = embed(AHEmat[:,41],13)
aheSDlags = embed(AHEmat[:,42],13)
aheTNlags = embed(AHEmat[:,43],13)
aheTXlags = embed(AHEmat[:,44],13)
aheUTlags = embed(AHEmat[:,45],13)
aheVTlags = embed(AHEmat[:,46],13)
aheVAlags = embed(AHEmat[:,47],13)
aheWAlags = embed(AHEmat[:,48],13)
aheWVlags = embed(AHEmat[:,49],13)
aheWIlags = embed(AHEmat[:,50],13)
aheWYlags = embed(AHEmat[:,51],13)

AHElaglong = [aheALlags; 	aheAKlags; 	aheAZlags; 	aheARlags; 	aheCAlags;
    aheCOlags; 	aheCTlags; 	aheDElags; 	aheDClags; 	aheFLlags; 	aheGAlags;
    aheHIlags; 	aheIDlags; 	aheILlags; 	aheINlags; 	aheIAlags; 	aheKSlags;
    aheKYlags; 	aheLAlags; 	aheMElags; 	aheMDlags; 	aheMAlags; 	aheMIlags;
    aheMNlags; 	aheMSlags; 	aheMOlags; 	aheMTlags; 	aheNElags; 	aheNVlags;
    aheNHlags; 	aheNJlags; 	aheNMlags; 	aheNYlags; 	aheNClags; 	aheNDlags;
    aheOHlags; 	aheOKlags; 	aheORlags; 	ahePAlags; 	aheRIlags; 	aheSClags;
    aheSDlags; 	aheTNlags; 	aheTXlags; 	aheUTlags; 	aheVTlags; 	aheVAlags;
    aheWAlags; 	aheWVlags; 	aheWIlags; 	aheWYlags]

yAHEpanel = AHElaglong[:,1]
xAHEpanel = AHElaglong[:,2:end]

MWmat = convert(Matrix, df2[:,104:154])
mwALlags = embed(MWmat[:,1],13)
mwAKlags = embed(MWmat[:,2],13)
mwAZlags = embed(MWmat[:,3],13)
mwARlags = embed(MWmat[:,4],13)
mwCAlags = embed(MWmat[:,5],13)
mwCOlags = embed(MWmat[:,6],13)
mwCTlags = embed(MWmat[:,7],13)
mwDElags = embed(MWmat[:,8],13)
mwDClags = embed(MWmat[:,9],13)
mwFLlags = embed(MWmat[:,10],13)
mwGAlags = embed(MWmat[:,11],13)
mwHIlags = embed(MWmat[:,12],13)
mwIDlags = embed(MWmat[:,13],13)
mwILlags = embed(MWmat[:,14],13)
mwINlags = embed(MWmat[:,15],13)
mwIAlags = embed(MWmat[:,16],13)
mwKSlags = embed(MWmat[:,17],13)
mwKYlags = embed(MWmat[:,18],13)
mwLAlags = embed(MWmat[:,19],13)
mwMElags = embed(MWmat[:,20],13)
mwMDlags = embed(MWmat[:,21],13)
mwMAlags = embed(MWmat[:,22],13)
mwMIlags = embed(MWmat[:,23],13)
mwMNlags = embed(MWmat[:,24],13)
mwMSlags = embed(MWmat[:,25],13)
mwMOlags = embed(MWmat[:,26],13)
mwMTlags = embed(MWmat[:,27],13)
mwNElags = embed(MWmat[:,28],13)
mwNVlags = embed(MWmat[:,29],13)
mwNHlags = embed(MWmat[:,30],13)
mwNJlags = embed(MWmat[:,31],13)
mwNMlags = embed(MWmat[:,32],13)
mwNYlags = embed(MWmat[:,33],13)
mwNClags = embed(MWmat[:,34],13)
mwNDlags = embed(MWmat[:,35],13)
mwOHlags = embed(MWmat[:,36],13)
mwOKlags = embed(MWmat[:,37],13)
mwORlags = embed(MWmat[:,38],13)
mwPAlags = embed(MWmat[:,39],13)
mwRIlags = embed(MWmat[:,40],13)
mwSClags = embed(MWmat[:,41],13)
mwSDlags = embed(MWmat[:,42],13)
mwTNlags = embed(MWmat[:,43],13)
mwTXlags = embed(MWmat[:,44],13)
mwUTlags = embed(MWmat[:,45],13)
mwVTlags = embed(MWmat[:,46],13)
mwVAlags = embed(MWmat[:,47],13)
mwWAlags = embed(MWmat[:,48],13)
mwWVlags = embed(MWmat[:,49],13)
mwWIlags = embed(MWmat[:,50],13)
mwWYlags = embed(MWmat[:,51],13)

MWlaglong =[mwALlags; 	mwAKlags; 	mwAZlags; 	mwARlags; 	mwCAlags;
    mwCOlags; 	mwCTlags; 	mwDElags; 	mwDClags; 	mwFLlags; 	mwGAlags;
    mwHIlags; 	mwIDlags; 	mwILlags; 	mwINlags; 	mwIAlags; 	mwKSlags;
    mwKYlags; 	mwLAlags; 	mwMElags; 	mwMDlags; 	mwMAlags; 	mwMIlags;
    mwMNlags; 	mwMSlags; 	mwMOlags; 	mwMTlags; 	mwNElags; 	mwNVlags;
    mwNHlags; 	mwNJlags; 	mwNMlags; 	mwNYlags; 	mwNClags; 	mwNDlags;
    mwOHlags; 	mwOKlags; 	mwORlags; 	mwPAlags; 	mwRIlags; 	mwSClags;
    mwSDlags; 	mwTNlags; 	mwTXlags; 	mwUTlags; 	mwVTlags; 	mwVAlags;
    mwWAlags; 	mwWVlags; 	mwWIlags; 	mwWYlags]

yMWpanel = MWlaglong[:,1]
xMWpanel = MWlaglong[:,2:end]

UERmat = convert(Matrix, df2[:,155:205])
uerALlags = embed(UERmat[:,1],13)
uerAKlags = embed(UERmat[:,2],13)
uerAZlags = embed(UERmat[:,3],13)
uerARlags = embed(UERmat[:,4],13)
uerCAlags = embed(UERmat[:,5],13)
uerCOlags = embed(UERmat[:,6],13)
uerCTlags = embed(UERmat[:,7],13)
uerDElags = embed(UERmat[:,8],13)
uerDClags = embed(UERmat[:,9],13)
uerFLlags = embed(UERmat[:,10],13)
uerGAlags = embed(UERmat[:,11],13)
uerHIlags = embed(UERmat[:,12],13)
uerIDlags = embed(UERmat[:,13],13)
uerILlags = embed(UERmat[:,14],13)
uerINlags = embed(UERmat[:,15],13)
uerIAlags = embed(UERmat[:,16],13)
uerKSlags = embed(UERmat[:,17],13)
uerKYlags = embed(UERmat[:,18],13)
uerLAlags = embed(UERmat[:,19],13)
uerMElags = embed(UERmat[:,20],13)
uerMDlags = embed(UERmat[:,21],13)
uerMAlags = embed(UERmat[:,22],13)
uerMIlags = embed(UERmat[:,23],13)
uerMNlags = embed(UERmat[:,24],13)
uerMSlags = embed(UERmat[:,25],13)
uerMOlags = embed(UERmat[:,26],13)
uerMTlags = embed(UERmat[:,27],13)
uerNElags = embed(UERmat[:,28],13)
uerNVlags = embed(UERmat[:,29],13)
uerNHlags = embed(UERmat[:,30],13)
uerNJlags = embed(UERmat[:,31],13)
uerNMlags = embed(UERmat[:,32],13)
uerNYlags = embed(UERmat[:,33],13)
uerNClags = embed(UERmat[:,34],13)
uerNDlags = embed(UERmat[:,35],13)
uerOHlags = embed(UERmat[:,36],13)
uerOKlags = embed(UERmat[:,37],13)
uerORlags = embed(UERmat[:,38],13)
uerPAlags = embed(UERmat[:,39],13)
uerRIlags = embed(UERmat[:,40],13)
uerSClags = embed(UERmat[:,41],13)
uerSDlags = embed(UERmat[:,42],13)
uerTNlags = embed(UERmat[:,43],13)
uerTXlags = embed(UERmat[:,44],13)
uerUTlags = embed(UERmat[:,45],13)
uerVTlags = embed(UERmat[:,46],13)
uerVAlags = embed(UERmat[:,47],13)
uerWAlags = embed(UERmat[:,48],13)
uerWVlags = embed(UERmat[:,49],13)
uerWIlags = embed(UERmat[:,50],13)
uerWYlags = embed(UERmat[:,51],13)

UERlaglong = [uerALlags; 	uerAKlags; 	uerAZlags; 	uerARlags; 	uerCAlags;
    uerCOlags; 	uerCTlags; 	uerDElags; 	uerDClags; 	uerFLlags; 	uerGAlags;
    uerHIlags; 	uerIDlags; 	uerILlags; 	uerINlags; 	uerIAlags; 	uerKSlags;
    uerKYlags; 	uerLAlags; 	uerMElags; 	uerMDlags; 	uerMAlags; 	uerMIlags;
    uerMNlags; 	uerMSlags; 	uerMOlags; 	uerMTlags; 	uerNElags; 	uerNVlags;
    uerNHlags; 	uerNJlags; 	uerNMlags; 	uerNYlags; 	uerNClags; 	uerNDlags;
    uerOHlags; 	uerOKlags; 	uerORlags; 	uerPAlags; 	uerRIlags; 	uerSClags;
    uerSDlags; 	uerTNlags; 	uerTXlags; 	uerUTlags; 	uerVTlags; 	uerVAlags;
    uerWAlags; 	uerWVlags; 	uerWIlags; 	uerWYlags]

yUERpanel = UERlaglong[:,1]
xUERpanel = UERlaglong[:,2:end]

py1 = yEPRpanel
py2 = yUERpanel
py3 = yAHEpanel
py4 = yMWpanel

panY = [yEPRpanel;yUERpanel;yAHEpanel;yMWpanel]
panX1 = [xEPRpanel xUERpanel xAHEpanel xMWpanel]


#number of variables (aka K) will be different if we include indicators
#indicator variables
dummymat = zeros(51,51)+I
dummyvec = ones(161) #length of one "wave"
stateindicators = kron(dummymat, dummyvec)

panX = [panX1 stateindicators]


# testpanelmodel = [EPRlaglong AHElaglong MWlaglong UERlaglong stateindicators]
# panY = testpanelmodel[:,1]
# panX = testpanelmodel[:,2:107]

#not working yet
# #export for use in other programs
# statapany = DataFrame(panY)
# CSV.write("statapanY.csv", panY)
# CSV.write("statapanX.csv", panX)

pnobs = 8211

#Define some terms that will be useful later when running the Gibbs
#sampler.

pbigX1 = [ones(size(panX)[1]) panX]

pX1X1 = pbigX1'*pbigX1
#no need to repeat since all X's are the same

#recall, that later on, it may call for X2Y1 (for example) and all X's are the same
pX1Y1 = pbigX1'*py1
pX1Y2 = pbigX1'*py2
pX1Y3 = pbigX1'*py3
pX1Y4 = pbigX1'*py4
pytilde = zeros(4,pnobs)
pxtilde = zeros(4,416,pnobs)
#ytilde: #eqns & nobs, xtilde:#eqns, total variables, nobs
for i in 1:pnobs
    pytilde[:,i] = [py1[i,1] py2[i,1] py3[i,1] py4[i,1]]'
    pxtilde[:,:,i] = [1 panX[i,:]' zeros(312)';
                    zeros(104)' 1 panX[i,:]' zeros(208)';
                    zeros(208)' 1 panX[i,:]' zeros(104)';
                    zeros(312)' 1 panX[i,:]']

    return pytilde, pxtilde
end

#Give the panel data TS priors
pβ0 = [TSβ1; zeros(204)]
#change B0 and β0 according to the number of (total) variables
pB0dg1 = diag(TSB1)
pB0dg2 = ones(204)*100
pB0dg3 = [pB0dg1; pB0dg2]
pB0 = Diagonal(pB0dg3) + zeros(416,416)
pR0 =  TSR1
pv0 = 0 #what are appropriate values for this?
pv1 = v0 + 104 #v0 + number of parameters

#Gibbs sampler
piter = 1000
pburn = 200
psig1_final = zeros(iter-burn,1)
psig2_final = zeros(iter-burn,1)
psig3_final = zeros(iter-burn,1)
psig4_final = zeros(iter-burn,1)
pcorr12_final = zeros(iter-burn,1)
pcorr13_final = zeros(iter-burn,1)
pcorr14_final = zeros(iter-burn,1)
pcorr23_final = zeros(iter-burn,1)
pcorr24_final = zeros(iter-burn,1)
pcorr34_final = zeros(iter-burn,1)
pbetas_final = zeros(iter-burn,416)
pB1_final = zeros(416,416)
#be sure to change betas_final to reflect no. of total variables

pSig1 = TSsig1
pSig2 = TSsig2
pSig3 = TSsig3
pSig4 = TSsig4
pSig12 = TSsig12
pSig13 = TSsig13
pSig14 = TSsig14
pSig23 = TSsig23
pSig24 = TSsig24
pSig34 = TSsig34

function psur_gibbs_ian(Sig1, Sig2, Sig3, Sig4, Sig12, Sig13, Sig14, Sig23, Sig24, Sig34, B0, β0, R0, v0)
    for i in 1:iter
    #for i in 1:5 #test loop
        #Sample from Conditional for Beta
        #note X1X1 stands in for X1X2, X1X3, etc since all X's are the same
        matrix1 = [Sig1*pX1X1 Sig12*pX1X1 Sig13*pX1X1 Sig14*pX1X1;
                   Sig12*pX1X1 Sig2*pX1X1 Sig23*pX1X1 Sig24*pX1X1;
                   Sig13*pX1X1 Sig23*pX1X1 Sig3*pX1X1 Sig34*pX1X1;
                   Sig14*pX1X1 Sig24*pX1X1 Sig34*pX1X1 Sig4*pX1X1]
        matrix2 = [ Sig1*pX1Y1 + Sig12*pX1Y2 + Sig13*pX1Y3 + Sig14*pX1Y4;
                    Sig12*pX1Y1 + Sig2*pX1Y2 + Sig23*pX1Y3 + Sig24*pX1Y4;
                    Sig13*pX1Y1 + Sig23*pX1Y2 + Sig3*pX1Y3 + Sig34*pX1Y4;
                    Sig14*pX1Y1 + Sig24*pX1Y2 + Sig34*pX1Y3 + Sig4*pX1Y4]
        B1 = inv(matrix1 + inv(B0))
        β1 = matrix2 + inv(B0)*β0
        B1herm = Symmetric(Hermitian(B1))+I*100 #adding to diagonal to make it sufficiently large for PSD
        #take the proceeding B1 and B1herm out and uncomment out the preceding B1herm
        #B1 = Matrix(Diagonal(B1)) #we did this to look at the diagonal elements when offdiags to zero
        #B1herm = Symmetric(Hermitian(B1))
        B1Sym = LinearAlgebra.cholesky(B1herm).U'*LinearAlgebra.cholesky(B1herm).U
        #carryover from code I found... what does this do?
        H_beta = LinearAlgebra.cholesky(Symmetric(B1Sym)).U
        betas = B1*β1 + H_beta'*randn(416,1)
            #randn(416,1) = total number of variables aka, one row of Xj

        #Sample from Wishart conditional for Sigmainv
        tempp = zeros(4,4)
        for j in 1:pnobs
            temp_term = (pytilde[:,j] - pxtilde[:,:,j]*betas)*(pytilde[:,j] - pxtilde[:,:,j]*betas)'
            tempp = temp_term + tempp
        end

        #inv(tempp + inv(R0)) is R1 in Greenberg
        R1 = inv(tempp + inv(R0))
        R1Sym = LinearAlgebra.cholesky(Hermitian(R1)).U'*LinearAlgebra.cholesky(Hermitian(R1)).U
        Sigmainv = rand(Wishart(nobs+v1, R1Sym))
        Sigma = inv(Sigmainv)
        Sig1 = Sigmainv[1,1]
        Sig2 = Sigmainv[2,2]
        Sig3 = Sigmainv[3,3]
        Sig4 = Sigmainv[4,4]
        Sig12 = Sigmainv[1,2]
        Sig13 = Sigmainv[1,3]
        Sig14 = Sigmainv[1,4]
        Sig23 = Sigmainv[2,3]
        Sig24 = Sigmainv[2,4]
        Sig34 = Sigmainv[3,4]

        s1 = Sigma[1,1]
        s2 = Sigma[2,2]
        s3 = Sigma[3,3]
        s4 = Sigma[4,4]
        s12 = Sigma[1,2]
        s13 = Sigma[1,3]
        s14 = Sigma[1,4]
        s23 = Sigma[2,3]
        s24 = Sigma[2,4]
        s34 = Sigma[3,4]

        #if i < burn #for testing
         if i > burn
             psig1_final[i-burn,1] = s1
             psig2_final[i-burn,1] = s2
             psig3_final[i-burn,1] = s3
             psig4_final[i-burn,1] = s4
             pcorr12_final[i-burn,1] = s12/(sqrt(s1)*sqrt(s2))
             pcorr13_final[i-burn,1] = s13/(sqrt(s1)*sqrt(s3))
             pcorr14_final[i-burn,1] = s14/(sqrt(s1)*sqrt(s4))
             pcorr23_final[i-burn,1] = s23/(sqrt(s2)*sqrt(s3))
             pcorr24_final[i-burn,1] = s24/(sqrt(s2)*sqrt(s4))
             pcorr34_final[i-burn,1] = s34/(sqrt(s3)*sqrt(s4))
             pbetas_final[i-burn,:] = betas'
             pB1_final[:,:] = B1
         end
    end

    return psig1_final, psig2_final, psig3_final, psig4_final, pcorr12_final, pcorr13_final, pcorr14_final, pcorr23_final, pcorr24_final, pcorr34_final, pbetas_final, pB1_final
end

poutput = psur_gibbs_ian(pSig1, pSig2, pSig3, pSig4,
    pSig12, pSig13, pSig14, pSig23, pSig24, pSig34,
    pB0, pβ0, pR0, pv0)


psig1 = mean(poutput[1])
psig2 = mean(poutput[2])
psig3 = mean(poutput[3])
psig4 = mean(poutput[4])
psig12 = mean(poutput[5])
psig13 = mean(poutput[6])
psig14 = mean(poutput[7])
psig23 = mean(poutput[8])
psig24 = mean(poutput[9])
psig34 = mean(poutput[10])
pβ1 = poutput[11]
pB1 = poutput[12]

TSβ1 = zeros(size(TSbeta)[2],1)
for i in 1:length(TSβ1)
    TSβ1[i] = mean(TSbeta[:,i])
    return TSβ1
end

TSR1 = [TSsig1 TSsig12 TSsig13 TSsig14;
        TSsig12 TSsig2 TSsig23 TSsig24;
        TSsig13 TSsig23 TSsig3 TSsig34;
        TSsig14 TSsig24 TSsig34 TSsig4]

#what do we do about PSD-ness? off-diagonal problem?
#
