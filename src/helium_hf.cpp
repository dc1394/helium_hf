/*! \file helium_hf.cpp
    \brief Hartree-Fock法で、ヘリウム原子のエネルギーを計算する
    Copyright © 2017 @dc1394 All Rights Reserved.
    (but this is originally adapted by Paolo Giannozzi for helium_hf_gauss.c from http://www.fisica.uniud.it/~giannozz/Corsi/MQ/Software/C/helium_hf_gauss.c )
    This software is released under the BSD 2-Clause License.
*/

#include <cmath>                                // for std::pow, std::sqrt
#include <cstdint>                              // for std::int32_t
#include <iostream>                             // for std::cout, std::cerr
#include <optional>                             // for std::make_optional, std::optional, std::nullopt
#include <vector>                               // for std::vector
#include <boost/assert.hpp>                     // for BOOST_ASSERT
#include <boost/format.hpp>                     // for boost::format
#include <boost/math/constants/constants.hpp>   // for boost::math::constants::pi
#include <boost/multi_array.hpp>                // for boost::multi_array
#include <Eigen/Core>                           // for Eigen::MatrixXd, Eigen::VectorXd
#include <Eigen/Eigenvalues>                    // for Eigen::GeneralizedSelfAdjointEigenSolver

namespace {
    //! A global variable (constant expression).
    /*!
        SCF計算のループの上限
    */
	static auto constexpr MAXITER = 1000;

    //! A global variable (constant expression).
    /*!
        Hartree-Fock計算に使用するGTOの個数
    */
	static auto constexpr NALPHA = 4;

    //! A global variable (constant expression).
    /*!
        SCF計算のループから抜ける際のエネルギーの差の閾値
    */
	static auto constexpr SCFTHRESHOLD = 1.0E-15;

    //! A global function.
    /*!
        SCF計算を行う
        \return SCF計算が正常に終了した場合はエネルギーを、しなかった場合はstd::nulloptを返す
    */
    std::optional<double> doscfloop();

    //! A global function.
    /*!
        ヘリウム原子のエネルギーを計算する
        \param c 固有ベクトルC
        \param ep 一般化固有値問題のエネルギー固有値E'
        \param h 1電子積分
        \return ヘリウム原子のエネルギー
    */
    double getenergy(Eigen::VectorXd const & c, double ep, boost::multi_array<double, 2> const & h);

    //! A global function.
    /*!
        GTOの肩の係数が格納された配列を生成する
        \param n GTOの個数
        \return GTOの肩の係数が格納されたstd::vector
    */
    std::vector<double> make_alpha(std::int32_t n);

    //! A global function.
    /*!
        全ての要素が、引数で指定された値で埋められたベクトルを生成する
        \param val 要素を埋める値
        \return 引数で指定された値で埋められたベクトル（Eigen::VectorXd）
    */
    Eigen::VectorXd make_c(double val);

    //! A global function.
    /*!
        与えられた固有ベクトル、1電子積分および2電子積分からFock行列を生成する
        \param c 固有ベクトルC
        \param h 1電子積分hpq
        \param q 2電子積分Qprqs
        \return Fock行列
    */
    Eigen::MatrixXd make_fockmatrix(Eigen::VectorXd const & c, boost::multi_array<double, 2> const & h, boost::multi_array<double, 4> const & q);

    //! A global function.
    /*!
        1電子積分が格納された2次元配列を生成する
        \param alpha GTOの肩の係数が格納されたstd::vector
        \return 1電子積分が格納された2次元配列（boost::multi_array）
    */
    boost::multi_array<double, 2> make_oneelectoroninteg(std::vector<double> const & alpha);
    
    //! A global function.
    /*!
        重なり行列を生成する
        \param alpha GTOの肩の係数が格納されたstd::vector
        \return 重なり行列（Eigen::MatrixXd）
    */
    Eigen::MatrixXd make_overlapmatrix(std::vector<double> const & alpha);
    
    //! A global function.
    /*!
        2電子積分が格納された4次元配列を生成する
        \param alpha GTOの肩の係数が格納されたstd::vector
        \return 2電子積分が格納された4次元配列（boost::multi_array）
    */
    boost::multi_array<double, 4> make_twoelectoroninteg(std::vector<double> const & alpha);
}

int main()
{
    if (auto const res = doscfloop(); res) {
        std::cout << boost::format("SCF計算が収束しました: energy = %.14f (Hartree)") % (*res) << std::endl;

        return 0;
    }
    else {
        std::cerr << "SCF計算が収束しませんでした" << std::endl;

        return -1;
    }

}

namespace {
    std::optional<double> doscfloop()
    {
        // GTOの肩の係数が格納された配列を生成
        auto alpha = make_alpha(NALPHA);

        // 1電子積分が格納された2次元配列を生成
        auto const h(make_oneelectoroninteg(alpha));

        // 2電子積分が格納された4次元配列を生成
        auto const q(make_twoelectoroninteg(alpha));

        // 重なり行列を生成
        auto const s(make_overlapmatrix(alpha));

        // 全て0.0で初期化された固有ベクトルを生成
        auto c(make_c(0.0));

        // 新しく計算されたエネルギー
        auto enew = 0.0;

        // SCFループ
        for (auto iter = 1; iter < MAXITER; iter++) {
            // Fock行列を生成
            auto const f(make_fockmatrix(c, h, q));

            // 一般化固有値問題を解く
            Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> es(f, s);
            
            // E'を取得
            auto const ep = es.eigenvalues()[0];
            
            // 固有ベクトルを取得
            c = es.eigenvectors().col(0);

            // 前回のSCF計算のエネルギーを保管
            auto const eold = enew;

            // 今回のSCF計算のエネルギーを計算する
            enew = getenergy(c, ep, h);

            std::cout << boost::format("Iteration # %2d: HF eigenvalue = %.14f, energy = %.14f\n") % iter % ep % enew;

            // SCF計算が収束したかどうか
            if (std::fabs(enew - eold) < SCFTHRESHOLD) {
                // 収束したのでそのエネルギーを返す
                return std::make_optional(enew);
            }
        }

        // SCF計算が収束しなかった
        return std::nullopt;
    }
    
    double getenergy(Eigen::VectorXd const & c, double ep, boost::multi_array<double, 2> const & h)
    {
        auto e = ep;
        for (auto p = 0; p < NALPHA; p++) {
            for (auto q = 0; q < NALPHA; q++) {
                // E = E' + Cp * Cq * hpq
                e += c[p] * c[q] * h[p][q];
            }
        }

        return e;
    }
    
    std::vector<double> make_alpha(std::int32_t n)
    {
        switch (n) {
        case 3:
            return { 0.31364978999999998, 1.1589229999999999, 6.3624213899999997 };
            break;

        case 4:
            return { 0.297104, 1.236745, 5.749982, 38.2166777 };
            break;

        case 6:
            return { 0.100112428, 0.24307674700000001, 0.62595526599999995, 1.8221429039999999, 6.5131437249999999, 35.523221220000003 };
            break;

        default:
            BOOST_ASSERT(!"switch文のdefaultに来てしまった！");
            return std::vector<double>();
            break;
        }
    }

    Eigen::VectorXd make_c(double val)
    {
        Eigen::VectorXd c(NALPHA);

        // 固有ベクトルCの要素を全てvalで初期化
        for (auto i = 0; i < NALPHA; i++) {
            c[i] = val;
        }

        return c;
    }

    Eigen::MatrixXd make_fockmatrix(Eigen::VectorXd const & c, boost::multi_array<double, 2> const & h, boost::multi_array<double, 4> const & q)
    {
        Eigen::MatrixXd f = Eigen::MatrixXd::Zero(NALPHA, NALPHA);

        for (auto p = 0; p < NALPHA; p++) {
            for (auto qi = 0; qi < NALPHA; qi++) {
                // Fpq = hpq + ΣCr * Cs * Qprqs
                f(p, qi) = h[p][qi];

                for (auto r = 0; r < NALPHA; r++) {
                    for (auto s = 0; s < NALPHA; s++) {
                        f(p, qi) += c[r] * c[s] * q[p][r][qi][s];
                    }
                }
            }
        }

        return f;
    }

    boost::multi_array<double, 2> make_oneelectoroninteg(std::vector<double> const & alpha)
    {
        using namespace boost::math::constants;

        boost::multi_array<double, 2> h(boost::extents[NALPHA][NALPHA]);

        for (auto p = 0; p < NALPHA; p++) {
            for (auto q = 0; q < NALPHA; q++) {
                // αp + αq
                auto const appaq = alpha[p] + alpha[q];

                // hpq = 3αpαqπ^1.5 / (αp + αq)^2.5 - 4π / (αp + αq)
                h[p][q] = 3.0 * alpha[p] * alpha[q] * std::pow((pi<double>() / appaq), 1.5) / appaq -
                          4.0 * pi<double>() / appaq;
            }
        }

        return h;
    }

    Eigen::MatrixXd make_overlapmatrix(std::vector<double> const & alpha)
    {
        using namespace boost::math::constants;

        Eigen::MatrixXd s = Eigen::MatrixXd::Zero(NALPHA, NALPHA);

        for (auto p = 0; p < NALPHA; p++) {
            for (auto q = 0; q < NALPHA; q++) {
                // Spq = (π / (αp + αq))^1.5
                s(p, q) = std::pow((pi<double>() / (alpha[p] + alpha[q])), 1.5);
            }
        }

        return s;
    }

    boost::multi_array<double, 4> make_twoelectoroninteg(std::vector<double> const & alpha)
    {
        using namespace boost::math::constants;

        boost::multi_array<double, 4> q(boost::extents[NALPHA][NALPHA][NALPHA][NALPHA]);

        for (auto p = 0; p < NALPHA; p++) {
            for (auto qi = 0; qi < NALPHA; qi++) {
                for (auto r = 0; r < NALPHA; r++) {
                    for (auto s = 0; s < NALPHA; s++) {
                        // Qprqs = 2π^2.5 / [(αp + αq)(αr + αs)√(αp + αq + αr + αs)]
                        q[p][r][qi][s] = 2.0 * std::pow(pi<double>(), 2.5) /
                            ((alpha[p] + alpha[qi]) * (alpha[r] + alpha[s]) *
                            std::sqrt(alpha[p] + alpha[qi] + alpha[r] + alpha[s]));
                    }
                }
            }
        }

        return q;
    }
}
